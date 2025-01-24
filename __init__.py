""""
AsyncCache Module
----------------------------

Copyright © HollowTheSilver 2024-2025 - https://github.com/HollowTheSilver

Version: 3.4.2

Description:

A high-performance asynchronous handler for nested data structures with advanced features
including path watching, batch operations, schema validation, transactions, and state management.
This module provides thread-safe operations for accessing and manipulating deeply nested data
structures with support for complex querying, validation, and transformation operations.

Key Features:
- Asynchronous operations with lock-based thread safety
- Path-based data access and modification with dot notation support
- Schema validation with customizable rules
- Transaction management with rollback capability
- State backup and restoration
- Pattern-based event subscription
- Batch operations for improved performance
- Deep merge capabilities
- Data transformation utilities

Example:
    ```python
    async def main():
        # Initialize handler with schema validation
        handler = AsyncDataHandler(
            {
                "users": {
                    "123": {"name": "John", "age": 30}
                }
            },
            schema={
                "type": "object",
                "properties": {
                    "users": {
                        "type": "object",
                        "properties": {
                            "*": {  # Wildcard for any user ID
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string"},
                                    "age": {"type": "integer"}
                                }
                            }
                        }
                    }
                }
            }
        )

        # Start a transaction
        async with handler.transaction():
            # Perform batch updates
            await handler.batch_set({
                "users.123.name": "John Doe",
                "users.123.age": 31
            })

            # Monitor changes
            async def on_change(value):
                print(f"User updated: {value}")
            await handler.subscribe("users.*", on_change)

            # Query data
            active_users = await handler.get(
                "users",
                query=lambda k, v: v.get("status") == "active"
            )
    ```
"""

# // ========================================( Modules )======================================== // #

import asyncio
import copy
import logging
import json
import re
import time
import uuid
from enum import Enum
from dataclasses import dataclass, asdict
from functools import lru_cache
from collections import defaultdict
from weakref import WeakValueDictionary
from typing import (
    TypeVar, Generic, Dict, List, Set, Union, Optional,
    Callable, Any, TypeAlias, cast, Tuple, Sequence,
    Generator, Awaitable, Literal, AsyncIterator,
    Iterator, TypedDict, TypeGuard, DefaultDict, Pattern
)


# // ========================================( Generics )======================================== // #


# Type variables for generic type support
T = TypeVar('T', bound=Dict[str, Any])  # Constrain T to be a dict
T_co = TypeVar('T_co', covariant=True)
K = TypeVar('K')
V = TypeVar('V')

# Type aliases for improved readability and type safety
PathType: TypeAlias = Union[str, Sequence[Union[str, int]]]
QueryType: TypeAlias = Callable[[Optional[str], Any], bool]
CallbackType: TypeAlias = Union[
    Callable[[Any], Awaitable[None]],
    Callable[[Any], None]
]
TransformType: TypeAlias = Callable[[Any], Any]
SchemaType: TypeAlias = Dict[str, Any]
CacheKeyType: TypeAlias = str
ValidationResultType: TypeAlias = bool
PathComponentType: TypeAlias = Union[str, int]
MergeStrategy = Literal["replace", "update", "deep"]


logger = logging.getLogger(__name__)


# // ========================================( Exceptions )======================================== // #


class NestedDataError(Exception):
    """Base exception class for AsyncNestedDataHandler errors."""
    pass


@dataclass
class PathError(NestedDataError):
    """
    Exception raised for invalid path operations.

    Attributes:
        path: The path that caused the error
        reason: Detailed explanation of the error
    """
    path: str
    reason: str

    def __str__(self) -> str:
        return f"Invalid path '{self.path}': {self.reason}"


@dataclass
class DataTypeError(NestedDataError):
    """
    Exception raised for type mismatches in data operations.

    Attributes:
        expected: Expected type
        received: Actual type received
        context: Optional context where the error occurred
    """
    expected: str
    received: str
    context: Optional[str] = None

    def __str__(self) -> str:
        msg = f"Expected {self.expected}, got {self.received}"
        if self.context:
            msg += f" at {self.context}"
        return msg


@dataclass
class ValidationError(NestedDataError):
    """
    Exception raised for schema validation errors.

    Attributes:
        path: Path where validation failed
        message: Detailed validation error message
    """
    path: str
    message: str

    def __str__(self) -> str:
        return f"Validation error at '{self.path}': {self.message}"


# // ========================================( Classes )======================================== // #


class ValidationCacheEntry(TypedDict):
    result: ValidationResultType
    access_count: int


@dataclass
class CacheEntry:
    """
    Represents a single cache entry with metadata for improved tracking and management.
    """
    result: ValidationResultType
    created_at: float
    last_accessed: float
    access_count: int
    data_hash: str
    schema_hash: str


class ValidationCache:
    """
    Optimized cache for schema validation results.
    """
    ValidatorType = Union[type, Callable[[Any], bool]]

    def __init__(
        self,
        max_size: int = 1000,
        cleanup_interval: float = 300,  # 5 minutes
        entry_timeout: float = 3600     # 1 hour
    ):
        self.max_size = max_size
        self.cleanup_interval = cleanup_interval
        self.entry_timeout = entry_timeout
        self._cache: Dict[CacheKeyType, ValidationResultType] = {}
        self._path_cache: Dict[str, Dict[str, ValidationResultType]] = {}
        self._access_count: Dict[CacheKeyType, int] = {}
        self._cache_lock = asyncio.Lock()
        self._last_cleanup = time.time()

    @staticmethod
    def _validate_type(value: Any, expected_type: str) -> bool:
        """
        Validate type with proper number handling.

        Args:
            value: The value to validate
            expected_type: String representing the expected type

        Returns:
            bool: True if the value matches the expected type, False otherwise
        """
        type_mapping: Dict[str, ValidationCache.ValidatorType] = {
            "string": str,
            "integer": int,
            "float": float,
            "number": (lambda x: isinstance(x, (int, float))),
            "boolean": bool,
            "array": list,
            "object": dict,
            "null": type(None)
        }
        validator = type_mapping.get(expected_type)
        if validator is None:
            return True
        return validator(value) if callable(validator) else isinstance(value, validator)

    @staticmethod
    def _hash_data(data: Any) -> str:
        """Create an efficient hash of the data for cache lookup."""
        if isinstance(data, (dict, list)):
            if isinstance(data, dict):
                items = list(data.items())[:3]
                return str(hash(f"d{len(data)}:{items}"))
            else:  # list
                items = data[:3]
                return str(hash(f"l{len(data)}:{items}"))
        return str(hash(data))

    def _generate_cache_key(self, data: Any, schema: dict, path: str = "") -> CacheKeyType:
        data_hash = self._hash_data(data)
        schema_hash = self._hash_data(schema)
        return f"{data_hash}:{schema_hash}:{path}"

    async def get_validation_result(
            self,
            data: Any,
            schema: dict,
            path: str = ""
    ) -> Optional[ValidationResultType]:
        """
        Thread-safe retrieval of cached validation results.

        Args:
            data: The data to validate
            schema: The schema to validate against
            path: The current path being validated

        Returns:
            Optional[ValidationResultType]: The cached validation result if available
        """
        cache_key = self._generate_cache_key(data, schema, path)

        async with self._cache_lock:
            if cache_key in self._cache:
                self._access_count[cache_key] = self._access_count.get(cache_key, 0) + 1
                return self._cache[cache_key]
            return None

    async def set_validation_result(
            self,
            data: Any,
            schema: dict,
            result: ValidationResultType,
            path: str = ""
    ) -> None:
        """
        Thread-safe storage of validation results.

        Args:
            data: The validated data
            schema: The schema used for validation
            result: The validation result
            path: The path that was validated
        """
        cache_key = self._generate_cache_key(data, schema, path)

        async with self._cache_lock:
            if len(self._cache) >= self.max_size:
                await self._perform_cleanup()

            self._cache[cache_key] = result
            self._access_count[cache_key] = 1

            if path:
                if path not in self._path_cache:
                    self._path_cache[path] = {}
                self._path_cache[path][self._hash_data(schema)] = result

    async def invalidate_path(self, path: str) -> None:
        """
        Invalidates cache entries for a path and its children.

        Args:
            path: The path to invalidate in the cache
        """
        async with self._cache_lock:
            # Find all paths that start with the given path
            paths_to_invalidate = [p for p in self._path_cache if p.startswith(path)]

            # Remove from path cache
            for p in paths_to_invalidate:
                if p in self._path_cache:
                    del self._path_cache[p]

            # Remove from main cache
            keys_to_remove = [
                k for k in self._cache
                if any(p in k for p in paths_to_invalidate)
            ]

            for k in keys_to_remove:
                del self._cache[k]
                if k in self._access_count:
                    del self._access_count[k]

    async def _perform_cleanup(self) -> None:
        """
        Removes expired and least recently used entries.
        """
        current_time = time.time()
        expired_keys = []

        for key, entry in self._cache.items():
            if (current_time - self._last_cleanup) > self.entry_timeout:
                expired_keys.append(key)

        for key in expired_keys:
            del self._cache[key]
            if key in self._access_count:
                del self._access_count[key]

        # If still over size limit, remove least accessed entries
        if len(self._cache) > self.max_size:
            sorted_entries = sorted(
                self._access_count.items(),
                key=lambda x: x[1]
            )
            remove_count = len(self._cache) - self.max_size
            for key, _ in sorted_entries[:remove_count]:
                if key in self._cache:
                    del self._cache[key]
                del self._access_count[key]

        self._last_cleanup = current_time

    async def clear(self) -> None:
        """Clear all cached validation results."""
        self._cache.clear()
        self._path_cache.clear()
        self._access_count.clear()


@dataclass
class TransactionEntry:
    """Represents a single change within a transaction"""
    path: str
    old_value: Any
    new_value: Any
    timestamp: float


class TransactionManager:
    """Enhanced transaction management system for AsyncDataHandler"""

    def __init__(self, handler: 'AsyncDataHandler'):
        self.handler = handler
        self._active_transaction: Optional[str] = None
        self._transaction_log: Dict[str, List[TransactionEntry]] = {}
        self._transaction_locks: Dict[str, Set[str]] = {}  # Track locked paths
        self._conflict_resolution_callbacks: Dict[str, callable] = {}
        self.lock = asyncio.Lock()

    async def begin_transaction(self, transaction_id: Optional[str] = None) -> str:
        """Begin a new transaction with optional custom ID"""
        async with self.lock:
            if self._active_transaction:
                raise ValueError("Transaction already in progress")

            transaction_id = transaction_id or f"txn_{len(self._transaction_log)}"
            self._active_transaction = transaction_id
            self._transaction_log[transaction_id] = []
            self._transaction_locks[transaction_id] = set()

            # Create state snapshot
            self.handler._transaction_stack.append(copy.deepcopy(self.handler.data))  # NOQA

            return transaction_id

    async def commit_transaction(self) -> None:
        """Commit the current transaction with conflict detection"""
        if not self._active_transaction:
            raise ValueError("No active transaction")

        async with self.lock:
            transaction_id = self._active_transaction

            # Check for conflicts with other committed transactions
            conflicts = await self._detect_conflicts(transaction_id)
            if conflicts:
                await self._handle_conflicts(conflicts)

            # Validate final state if needed
            if self.handler._auto_validate and self.handler._schema:
                if not await self.handler.validate():
                    await self.rollback_transaction()
                    raise ValueError("Schema validation failed")

            # Clear transaction data
            self._transaction_locks.pop(transaction_id, None)
            self._active_transaction = None
            self.handler._transaction_stack.pop()

            # Clear relevant cache entries
            if self.handler._validation_cache is not None:
                modified_paths = {entry.path for entry in self._transaction_log[transaction_id]}
                for path in modified_paths:
                    await self.handler._validation_cache.invalidate_path(path)

    async def rollback_transaction(self) -> None:
        """Rollback the current transaction"""
        if not self._active_transaction:
            raise ValueError("No active transaction")

        async with self.lock:
            transaction_id = self._active_transaction

            # Restore original state
            self.handler.data = self.handler._transaction_stack.pop()

            # Clear transaction data
            self._transaction_locks.pop(transaction_id, None)
            self._transaction_log.pop(transaction_id, None)
            self._active_transaction = None

            # Clear cache
            if self.handler._validation_cache is not None:
                await self.handler._validation_cache.clear()

    async def _detect_conflicts(self, transaction_id: str) -> List[Dict[str, Any]]:
        """Detect conflicts with other transactions"""
        conflicts = []
        current_changes = self._transaction_log[transaction_id]

        for other_id, other_changes in self._transaction_log.items():
            if other_id == transaction_id:
                continue

            for current_entry in current_changes:
                for other_entry in other_changes:
                    if (current_entry.path == other_entry.path and
                            current_entry.timestamp > other_entry.timestamp):
                        conflicts.append({
                            'path': current_entry.path,
                            'current_value': current_entry.new_value,
                            'conflicting_value': other_entry.new_value,
                            'transaction_id': other_id
                        })

        return conflicts

    async def _handle_conflicts(self, conflicts: List[Dict[str, Any]]) -> None:
        """Handle detected conflicts using registered resolution strategies"""
        for conflict in conflicts:
            path = conflict['path']
            if path in self._conflict_resolution_callbacks:
                resolver = self._conflict_resolution_callbacks[path]
                resolved_value = await resolver(
                    conflict['current_value'],
                    conflict['conflicting_value']
                )
                await self.handler.set(path, resolved_value)
            else:
                # Default to raising an error if no resolution strategy is defined
                raise ValueError(f"Unresolved conflict at path: {path}")

    async def register_conflict_resolver(
            self,
            path: str,
            resolver: callable
    ) -> None:
        """Register a conflict resolution callback for a specific path"""
        self._conflict_resolution_callbacks[path] = resolver

    async def lock_path(self, path: str) -> None:
        """Lock a path for exclusive access in the current transaction"""
        if not self._active_transaction:
            raise ValueError("No active transaction")

        async with self.lock:
            # Check if path is locked by another transaction
            for txn_id, locked_paths in self._transaction_locks.items():
                if txn_id != self._active_transaction and path in locked_paths:
                    raise ValueError(f"Path {path} is locked by another transaction")

            self._transaction_locks[self._active_transaction].add(path)

    async def unlock_path(self, path: str) -> None:
        """Unlock a previously locked path"""
        if not self._active_transaction:
            raise ValueError("No active transaction")

        async with self.lock:
            self._transaction_locks[self._active_transaction].discard(path)


class TransactionStatus(Enum):
    STARTED = "started"
    COMMITTED = "committed"
    ROLLED_BACK = "rolled_back"
    FAILED = "failed"


@dataclass
class TransactionMetrics:
    """Stores metrics for a single transaction"""
    transaction_id: str
    start_time: float
    end_time: Optional[float] = None
    status: TransactionStatus = TransactionStatus.STARTED
    modified_paths: List[str] = None
    conflicts_detected: int = 0
    conflicts_resolved: int = 0
    validation_attempts: int = 0
    error_message: Optional[str] = None

    def __post_init__(self):
        if self.modified_paths is None:
            self.modified_paths = []

    @property
    def duration(self) -> Optional[float]:
        """Calculate transaction duration in seconds"""
        if self.end_time is None:
            return None
        return self.end_time - self.start_time


class TransactionMonitor:
    """Monitors and tracks transaction metrics"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._active_transactions: Dict[str, TransactionMetrics] = {}
        self._completed_transactions: List[TransactionMetrics] = []
        self._max_history = 1000  # Keep last 1000 completed transactions
        self._monitor_lock = asyncio.Lock()

    async def start_transaction(self, transaction_id: str) -> None:
        """Record the start of a new transaction"""
        async with self._monitor_lock:
            metrics = TransactionMetrics(
                transaction_id=transaction_id,
                start_time=time.time()
            )
            self._active_transactions[transaction_id] = metrics
            self.logger.info(f"Transaction started: {transaction_id}")

    async def complete_transaction(
            self,
            transaction_id: str,
            status: TransactionStatus,
            error: Optional[str] = None
    ) -> None:
        """Record the completion of a transaction"""
        async with self._monitor_lock:
            if transaction_id not in self._active_transactions:
                self.logger.warning(f"Unknown transaction completed: {transaction_id}")
                return

            metrics = self._active_transactions.pop(transaction_id)
            metrics.end_time = time.time()
            metrics.status = status
            metrics.error_message = error

            self._completed_transactions.append(metrics)
            while len(self._completed_transactions) > self._max_history:
                self._completed_transactions.pop(0)

            self.logger.info(
                f"Transaction {transaction_id} {status.value} "
                f"(duration: {metrics.duration:.3f}s)"
            )

    async def record_conflict(self, transaction_id: str, was_resolved: bool) -> None:
        """Record a conflict detection or resolution"""
        async with self._monitor_lock:
            if transaction_id in self._active_transactions:
                metrics = self._active_transactions[transaction_id]
                metrics.conflicts_detected += 1
                if was_resolved:
                    metrics.conflicts_resolved += 1

    async def record_validation(self, transaction_id: str) -> None:
        """Record a validation attempt"""
        async with self._monitor_lock:
            if transaction_id in self._active_transactions:
                self._active_transactions[transaction_id].validation_attempts += 1

    async def record_modified_path(self, transaction_id: str, path: str) -> None:
        """Record a path modified by the transaction"""
        async with self._monitor_lock:
            if transaction_id in self._active_transactions:
                metrics = self._active_transactions[transaction_id]
                if path not in metrics.modified_paths:
                    metrics.modified_paths.append(path)

    async def get_active_transactions(self) -> List[TransactionMetrics]:
        """Get metrics for all active transactions"""
        async with self._monitor_lock:
            return list(self._active_transactions.values())

    async def get_completed_transactions(
            self,
            limit: Optional[int] = None,
            status_filter: Optional[TransactionStatus] = None
    ) -> List[TransactionMetrics]:
        """Get metrics for completed transactions with optional filtering"""
        async with self._monitor_lock:
            transactions = self._completed_transactions
            if status_filter:
                transactions = [t for t in transactions if t.status == status_filter]
            if limit:
                transactions = transactions[-limit:]
            return transactions

    async def get_transaction_metrics(
            self,
            transaction_id: str
    ) -> Optional[TransactionMetrics]:
        """Get metrics for a specific transaction"""
        async with self._monitor_lock:
            if transaction_id in self._active_transactions:
                return self._active_transactions[transaction_id]
            for transaction in self._completed_transactions:
                if transaction.transaction_id == transaction_id:
                    return transaction
            return None


@dataclass
class RecoveryEntry:
    """Represents a transaction entry for recovery purposes"""
    transaction_id: str
    timestamp: float
    data_snapshot: Dict
    modified_paths: List[str]
    status: str


class TransactionRecovery:
    """Handles transaction recovery after system failures"""

    def __init__(self, handler: 'AsyncDataHandler'):
        self.handler = handler
        self.logger = logging.getLogger(__name__)
        self._recovery_lock = asyncio.Lock()
        self._persistence_lock = asyncio.Lock()
        self._pending_transactions: Dict[str, RecoveryEntry] = {}

    async def save_recovery_point(self, transaction_id: str) -> None:
        """Save a recovery point for a transaction"""
        async with self._recovery_lock:
            recovery_entry = RecoveryEntry(
                transaction_id=transaction_id,
                timestamp=time.time(),
                data_snapshot=self.handler.data.copy(),
                modified_paths=await self._get_modified_paths(transaction_id),
                status="pending"
            )
            self._pending_transactions[transaction_id] = recovery_entry
            await self._persist_recovery_state()

    async def clear_recovery_point(self, transaction_id: str) -> None:
        """Clear a transaction's recovery point after successful completion"""
        async with self._recovery_lock:
            if transaction_id in self._pending_transactions:
                del self._pending_transactions[transaction_id]
                await self._persist_recovery_state()

    async def recover_pending_transactions(self) -> None:
        """Recover any pending transactions after system restart"""
        async with self._recovery_lock:
            pending = await self._load_recovery_state()
            if not pending:
                return

            self.logger.info(f"Found {len(pending)} pending transactions to recover")
            for transaction_id, entry in pending.items():
                try:
                    await self._recover_transaction(entry)
                except Exception as e:
                    self.logger.error(
                        f"Failed to recover transaction {transaction_id}: {str(e)}"
                    )

    async def _recover_transaction(self, entry: RecoveryEntry) -> None:
        """Recover a single transaction"""
        self.logger.info(f"Recovering transaction: {entry.transaction_id}")

        # Store current state for potential rollback
        original_state = self.handler.data.copy()

        try:
            # Temporarily apply recovered state
            self.handler.data = entry.data_snapshot

            # Validate the recovered state
            if self.handler._auto_validate and self.handler._schema:
                is_valid = await self.handler.validate()
                if not is_valid:
                    self.logger.error(
                        f"Recovered data failed validation for transaction "
                        f"{entry.transaction_id}"
                    )
                    self.handler.data = original_state
                    return

            # Apply the recovered changes within a transaction
            async with self.handler.transaction():
                await self._notify_recovery(entry.transaction_id, entry.modified_paths)

        except Exception as e:
            self.logger.error(
                f"Error applying recovered transaction {entry.transaction_id}: {str(e)}"
            )
            self.handler.data = original_state
            raise

    async def _notify_recovery(
            self,
            transaction_id: str,
            modified_paths: List[str]
    ) -> None:
        """Notify watchers of recovered changes"""
        for path in modified_paths:
            value = await self.handler.get(path)
            await self.handler._notify_watchers(path, value)

    async def _get_modified_paths(self, transaction_id: str) -> List[str]:
        """Get list of paths modified by a transaction"""
        if hasattr(self.handler._transaction_manager, '_transaction_log'):
            log = self.handler._transaction_manager._transaction_log.get(transaction_id, [])
            return list(set(entry.path for entry in log))
        return []

    async def _persist_recovery_state(self) -> None:
        """Persist recovery state to storage"""
        async with self._persistence_lock:
            state = {}
            for tid, entry in self._pending_transactions.items():
                try:
                    state[tid] = {
                        'transaction_id': entry.transaction_id,
                        'timestamp': entry.timestamp,
                        'data_snapshot': entry.data_snapshot,
                        'modified_paths': entry.modified_paths,
                        'status': entry.status
                    }
                except Exception as e:
                    self.logger.error(f"Error serializing recovery entry: {str(e)}")
                    continue

            try:
                # Convert state to JSON
                json_state = json.dumps(state)

                # In a production environment, you would want to save this to a
                # persistent storage (database, file system, etc.)
                # For example, to save to a file:
                # async with aiofiles.open('recovery_state.json', 'w') as f:
                #     await f.write(json_state)

                # For now, we'll just log it
                self.logger.debug(f"Would persist state: {json_state}")

            except Exception as e:
                self.logger.error(f"Failed to persist recovery state: {str(e)}")

    async def _load_recovery_state(self) -> Dict[str, RecoveryEntry]:
        """Load recovery state from storage"""
        async with self._persistence_lock:
            # Implementation would depend on your storage mechanism
            # Here is where you would load the state, e.g.:
            # return await self._load_from_storage()
            return {}


@dataclass
class SubscriptionMetadata:
    """
    Stores metadata for a subscription with proper typing.
    """
    pattern: str
    callback: CallbackType
    pattern_regex: Pattern
    include_child_paths: bool
    created_at: float
    last_triggered: float
    trigger_count: int


class AsyncDataHandler(Generic[T]):
    """
    A high-performance asynchronous handler for nested data structures.

    This class provides a comprehensive interface for working with nested data structures
    in an asynchronous context, ensuring thread safety and data consistency through
    various operations including schema validation, transactions, and event handling.

    Attributes:
        data: The nested data structure being managed
        default_return_type: Default return type for queries
        default_deep_search: Whether deep search is enabled by default
        lock: Asyncio lock for thread safety
        watchers: Dictionary mapping paths to callback functions
    """

    def __init__(
            self,
            data: T,
            return_type: Literal["value", "list", "generator"] = "value",
            deep_search: bool = False,
            cache_size: int = 128,
            auto_validate: bool = False,
            schema: Optional[SchemaType] = None,
            backup_enabled: bool = False,
            max_backup_count: int = 5
    ) -> None:
        """
        Initialize the AsyncNestedDataHandler.

        Args:
            data: Initial data structure (must be a dict)
            return_type: Default return type for queries ("value", "list", "generator")
            deep_search: Enable deep search by default
            cache_size: Size of the LRU cache for path parsing
            auto_validate: Enable automatic schema validation
            schema: Optional schema for validation
            backup_enabled: Enable state backups
            max_backup_count: Maximum number of backups to keep

        Raises:
            DataTypeError: If initial data is not a dict
            ValueError: If return_type is invalid
        """
        if not isinstance(data, dict):
            raise DataTypeError("dict", type(data).__name__)
        # Runtime validation maintained alongside type hints for additional safety
        if return_type not in {"value", "list", "generator"}:
            raise ValueError(f"Invalid return_type '{return_type}'")

        # Core data structure and schema validation
        self.data = data
        self._schema = schema
        self._auto_validate = auto_validate
        self._validation_cache: Optional[ValidationCache] = (
            ValidationCache() if auto_validate else None
        )

        # Query and operation settings
        self.default_return_type = return_type
        self.default_deep_search = deep_search
        self._cache_size = cache_size

        # Concurrency control
        self.lock = asyncio.Lock()
        self._notification_lock = asyncio.Lock()

        # Backup management
        self._backup_enabled = backup_enabled
        self._max_backup_count = max_backup_count
        self._backup_history: List[Tuple[str, T]] = []  # [(id, data), ...]

        # Transaction management
        self._transaction_stack: List[T] = []
        self._transaction_manager = TransactionManager(self)

        # Transaction monitoring and recovery
        self._transaction_monitor = TransactionMonitor()
        self._transaction_recovery = TransactionRecovery(self)

        # Event handling and subscriptions
        self.watchers: DefaultDict[str, List[CallbackType]] = defaultdict(list)
        self._active_notifications: Set[str] = set()
        self._subscriptions: WeakValueDictionary[str, SubscriptionMetadata] = WeakValueDictionary()
        self._pattern_cache: Dict[str, Pattern] = {}

    class Transaction(Generic[T_co]):
        """
        Context manager for transaction handling.

        Provides a convenient way to group operations and ensure they are either
        all completed successfully or rolled back in case of an error.

        Example:
            ```python
            async with handler.transaction():  # Changed from handler.Transaction()
                await handler.set("users.123.name", "John")
                await handler.set("users.123.age", 30)
            ```
        """

        def __init__(self, handler: 'AsyncDataHandler[T_co]'):
            self.handler = handler

        async def __aenter__(self) -> 'AsyncDataHandler[T_co]':
            await self.handler.begin_transaction()
            return self.handler

        async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
            if exc_type is None:
                await self.handler.commit_transaction()
            else:
                await self.handler.rollback_transaction()

    @property
    def transaction(self) -> 'Transaction':
        """
        Property that returns the Transaction context manager.

        Returns:
            Transaction context manager instance.

        Example:
            ```python
            async with handler.transaction():  # Changed from handler.Transaction()
                await handler.set("users.123.name", "John")
                await handler.set("users.123.age", 30)
            ```
        """
        return AsyncDataHandler.Transaction(self)

    @staticmethod
    @lru_cache(maxsize=128)
    def _parse_path(path: PathType) -> Tuple[PathComponentType, ...]:
        """
        Parse and cache path resolution for improved performance.

        Args:
            path: Path to parse (string or list format)

        Returns:
            Tuple of path components for internal use

        Raises:
            PathError: If path format is invalid
        """
        if isinstance(path, str):
            return tuple(path.split('.'))
        elif isinstance(path, (list, tuple)):
            return tuple(str(component) for component in path)
        raise PathError(str(path), "Path must be a string or sequence")

    async def _traverse(  # NOQA
            self,
            data: Any,
            keys: Tuple[PathComponentType, ...],
            create_missing: bool = False
    ) -> Tuple[Union[Dict[str, Any], List[Any]], Optional[PathComponentType]]:
        """
        Traverse the nested structure following the given keys.

        Args:
            data: The data structure to traverse
            keys: Tuple of keys to follow
            create_missing: Whether to create missing intermediate paths

        Returns:
            Tuple of (parent, key) where parent is the containing object
            and key is the final key in the path

        Raises:
            PathError: If path cannot be traversed
        """
        if not keys:
            return data, None

        current = data
        for key in keys[:-1]:
            if isinstance(current, dict):
                if key not in current:
                    if create_missing:
                        current[key] = {}
                    else:
                        raise PathError(str(key), "Key not found")
                current = current[key]
            elif isinstance(current, list):
                try:
                    idx = int(key)
                    while len(current) <= idx and create_missing:
                        current.append({})
                    current = current[idx]
                except (ValueError, IndexError) as e:
                    raise PathError(str(key), f"Invalid list index: {str(e)}")
            else:
                raise PathError(str(key), "Cannot traverse further")

        return current, keys[-1]

    async def get(
            self,
            path: Optional[PathType] = None,
            query: Optional[QueryType] = None,
            return_type: Optional[str] = None,
            default: Any = None,
            deep_search: Optional[bool] = None
    ) -> Any:
        """
        Retrieve data from the nested structure using path or query.

        This method provides flexible data retrieval with support for deep searching,
        custom queries, and different return types. It can traverse nested structures
        using dot notation or list indices.

        Args:
            path: Path to the data (dot notation or list)
            query: Optional query function for filtering results
            return_type: Override default return type
            default: Default value if path not found
            deep_search: Override default deep search setting

        Returns:
            Retrieved data or default value

        Raises:
            PathError: If path format is invalid

        Example:
            ```python
            # Simple path access
            user = await handler.get("users.123")

            # With query
            active_users = await handler.get(
                "users",
                query=lambda k, v: v.get("status") == "active"
            )

            # Deep search with custom return type
            results = await handler.get(
                deep_search=True,
                query=lambda k, v: isinstance(v, dict) and v.get("type") == "admin",
                return_type="list"
            )
            ```
        """
        deep_search = deep_search if deep_search is not None else self.default_deep_search
        return_type = return_type or self.default_return_type

        async with self.lock:
            target = self.data

            if path:
                try:
                    parent, key = await self._traverse(target, self._parse_path(path))
                    if isinstance(parent, (dict, list)):
                        try:
                            target = parent[key] if key is not None else parent
                        except (KeyError, IndexError):
                            return default
                    else:
                        return default
                except PathError:
                    return default

            if target is None:
                return default

            if query is None:
                return target

            # Apply query filtering with deep search if enabled
            def _search(data: Any) -> Generator[Any, None, None]:
                if isinstance(data, dict):
                    if query(None, data):
                        yield data
                    if deep_search:
                        for k, v in data.items():
                            if query(k, v):
                                yield v
                            yield from _search(v)
                elif isinstance(data, list):
                    for item in data:
                        yield from _search(item)
                elif query(None, data):
                    yield data

            results = list(_search(target))

            if return_type == "generator":
                return (x for x in results)
            elif return_type == "list":
                return results
            else:  # return_type == "value"
                return results[0] if results else default

    async def set(
            self,
            path: PathType,
            value: Any,
            create_missing: bool = True
    ) -> None:
        """
        Set a value at the specified path.

        Supports creating intermediate paths and maintains schema validation
        if enabled. Notifies watchers of changes.

        Args:
            path: Path where to set the value
            value: Value to set
            create_missing: Create missing intermediate paths

        Raises:
            PathError: If path is invalid or cannot be created
            ValidationError: If schema validation fails and auto_validate is True

        Example:
            ```python
            # Simple set
            await handler.set("users.123.name", "John Doe")

            # Create intermediate paths
            await handler.set("users.123.settings.notifications", True)

            # Set in list
            await handler.set("users.123.roles.0", "admin")
            ```
        """
        async with self.lock:
            keys = self._parse_path(path)
            parent, key = await self._traverse(self.data, keys, create_missing)

            if isinstance(parent, dict):
                parent[key] = value
            elif isinstance(parent, list):
                try:
                    idx = int(key)
                    while len(parent) <= idx and create_missing:
                        parent.append(None)
                    parent[idx] = value
                except (ValueError, IndexError) as e:
                    raise PathError(str(key), f"Invalid list index: {str(e)}")
            else:
                raise PathError(str(key), "Cannot set value at path")

            if self._validation_cache is not None:
                await self._validation_cache.invalidate_path(str(path))

            if self._auto_validate and self._schema:
                if not await self.validate():
                    raise ValidationError("validation", "Schema validation failed")

            path_str = '.'.join(map(str, self._parse_path(path)))
            watchers = self.watchers.get(path_str, [])[:]

            for callback in watchers:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(value)
                    else:
                        callback(value)
                except Exception as e:
                    logger.error(f"Error in watcher callback: {str(e)}")

    async def remove(
            self,
            path: PathType,
            silent: bool = False
    ) -> None:
        """
        Remove a value at the specified path.

        Args:
            path: Path to the value to remove
            silent: If True, don't notify watchers

        Raises:
            PathError: If path is invalid
            ValidationError: If removal would violate schema and auto_validate is True

        Example:
            ```python
            # Remove a user
            await handler.remove("users.123")

            # Remove silently
            await handler.remove("users.123.temp_data", silent=True)
            ```
        """
        async with self.lock:
            keys = self._parse_path(path)
            parent, key = await self._traverse(self.data, keys)

            if isinstance(parent, dict):
                if key in parent:
                    del parent[key]
                else:
                    raise PathError(str(key), "Key not found")
            elif isinstance(parent, list):
                try:
                    idx = int(key)
                    del parent[idx]
                except (ValueError, IndexError) as e:
                    raise PathError(str(key), f"Invalid list index: {str(e)}")
            else:
                raise PathError(str(key), "Cannot remove value at path")

            if self._validation_cache is not None:
                await self._validation_cache.invalidate_path(str(path))

            if self._auto_validate and self._schema:
                if not await self.validate():
                    raise ValidationError("validation", "Schema validation failed")

            if not silent:
                await self._notify_watchers(path, None)

    async def subscribe(
            self,
            pattern: str,
            callback: CallbackType,
            include_child_paths: bool = False,
            timeout: Optional[float] = None
    ) -> str:
        """
        Enhanced subscription method with proper error handling and subscription tracking.

        Args:
            pattern: Regex pattern to match against paths
            callback: Function to be called when matching paths change
            include_child_paths: Whether to include changes to child paths
            timeout: Optional timeout for subscription operations in seconds

        Returns:
            str: Subscription ID for later un-subscription

        Raises:
            ValueError: If the pattern is invalid
            TimeoutError: If the subscription operation times out
            RuntimeError: If subscription limit is reached

        Example:
            ```python
            # Subscribe to all user updates
            await handler.subscribe("users.*", lambda value: print(f"User updated: {value}"))

            # Watch specific user's name changes
            await handler.subscribe("users.123.name", lambda value: print(f"Name changed to: {value}"))

            # Watch all nested settings changes
            await handler.subscribe("users.*.settings.*",
                lambda value: print(f"Settings updated: {value}"),
                include_child_paths=True)
            ```
        """
        # Generate subscription ID
        subscription_id = str(uuid.uuid4())

        # Get or compile pattern with caching
        if pattern not in self._pattern_cache:
            try:
                self._pattern_cache[pattern] = re.compile(pattern)
            except re.error as e:
                raise ValueError(f"Invalid pattern '{pattern}': {str(e)}")

        pattern_regex = self._pattern_cache[pattern]

        async with self.lock:
            try:
                async with asyncio.timeout(timeout) if timeout else asyncio.TaskGroup() as tg:
                    path_str = '.'.join(map(str, self._parse_path(pattern)))

                    # Check subscription limits
                    if len(self._subscriptions) >= 1000:  # Configurable limit
                        await self._cleanup_expired_subscriptions()
                        if len(self._subscriptions) >= 1000:
                            raise RuntimeError("Maximum subscription limit reached")

                    # Create subscription metadata
                    subscription = SubscriptionMetadata(
                        pattern=pattern,
                        callback=callback,
                        pattern_regex=pattern_regex,
                        include_child_paths=include_child_paths,
                        created_at=time.time(),
                        last_triggered=time.time(),
                        trigger_count=0
                    )

                    # Add callback to watchers
                    self.watchers[path_str].append(callback)

                    if include_child_paths:
                        all_paths = set(self.watchers.keys())
                        for path in all_paths:
                            if any(pattern_regex.match(parent) for parent in path.split('.')):
                                if callback not in self.watchers[path]:
                                    self.watchers[path].append(callback)

                    # Store subscription metadata
                    self._subscriptions[subscription_id] = subscription

                    return subscription_id

            except asyncio.TimeoutError:
                logger.error(f"Subscription operation timed out for pattern: {pattern}")
                # Ensure cleanup in case of timeout
                tg.create_task(self._cleanup_subscription(subscription_id))
                raise
            except Exception as e:
                logger.error(f"Subscription failed for pattern '{pattern}': {str(e)}")
                tg.create_task(self._cleanup_subscription(subscription_id))
                raise

    async def unsubscribe(
            self,
            pattern: str,
            callback: Optional[CallbackType] = None
    ) -> None:
        """
        Remove subscription(s) for a pattern.

        Args:
            pattern: Pattern to unsubscribe from
            callback: Specific callback to remove, or None to remove all

        Example:
            ```python
            # Remove specific callback
            await handler.unsubscribe("users.*", callback_function)

            # Remove all callbacks for pattern
            await handler.unsubscribe("users.*")
            ```
        """
        async with self.lock:
            path_str = '.'.join(map(str, self._parse_path(pattern)))
            if callback is None:
                self.watchers[path_str].clear()
            else:
                self.watchers[path_str] = [
                    cb for cb in self.watchers[path_str]
                    if cb != callback
                ]

    async def _cleanup_expired_subscriptions(self) -> None:
        """
        Remove subscriptions that haven't been triggered recently.
        """
        current_time = time.time()
        expired_threshold = current_time - (24 * 60 * 60)  # 24 hours

        subscription_ids = list(self._subscriptions.keys())
        for subscription_id in subscription_ids:
            subscription = self._subscriptions.get(subscription_id)
            if subscription and subscription.last_triggered < expired_threshold:
                await self._cleanup_subscription(subscription_id)

    async def _cleanup_subscription(self, subscription_id: str) -> None:
        """
        Clean up subscription state in case of errors or timeouts.

        Args:
            subscription_id: The ID of the subscription to clean up
        """
        if subscription_id in self._subscriptions:
            subscription = self._subscriptions[subscription_id]

            async with self.lock:
                # Remove from watchers
                for path in self.watchers:
                    if subscription.callback in self.watchers[path]:
                        self.watchers[path].remove(subscription.callback)

                # Clean up empty watcher lists
                empty_paths = [path for path, callbacks in self.watchers.items() if not callbacks]
                for path in empty_paths:
                    del self.watchers[path]

                # Remove subscription metadata
                del self._subscriptions[subscription_id]

    async def _notify_watchers(self, path: str, value: Any) -> None:
        """
        Thread-safe notification of watchers with proper error handling.

        Args:
            path: The path where the change occurred
            value: The new value at the path
        """
        path_str = '.'.join(map(str, self._parse_path(path)))

        async with self._notification_lock:
            if path_str in self._active_notifications:
                return

            self._active_notifications.add(path_str)
            current_watchers = [(subscription_id, subscription.callback)
                                for subscription_id, subscription
                                in self._subscriptions.items()
                                if subscription.callback in self.watchers.get(path_str, [])]

        try:
            for subscription_id, callback in current_watchers:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await asyncio.wait_for(callback(value), timeout=5.0)
                    else:
                        callback(value)

                    # Update subscription metadata
                    if subscription_id in self._subscriptions:
                        subscription = self._subscriptions[subscription_id]
                        subscription.last_triggered = time.time()
                        subscription.trigger_count += 1

                except asyncio.TimeoutError:
                    logger.error(f"Watcher callback timed out for path: {path_str}")
                except Exception as e:
                    logger.error(f"Error in watcher callback for path {path_str}: {str(e)}")
        finally:
            async with self._notification_lock:
                self._active_notifications.remove(path_str)

    async def begin_transaction(self) -> None:
        """
        Begin a new transaction.

        Creates a snapshot of the current state that can be restored if the
        transaction needs to be rolled back.

        Raises:
            NestedDataError: If a transaction is already in progress

        Example:
            ```python
            await handler.begin_transaction()
            try:
                await handler.set("users.123.status", "inactive")
                await handler.commit_transaction()
            except:
                await handler.rollback_transaction()
            ```
        """
        if self._transaction_stack:
            raise NestedDataError("Transaction already in progress")

        async with self.lock:
            self._transaction_stack.append(copy.deepcopy(self.data))

    async def commit_transaction(self) -> None:
        """
        Commit the current transaction.

        Finalizes the changes made during the transaction. If auto_validate
        is enabled, validates the final state before committing.

        Raises:
            NestedDataError: If no transaction is active
            ValidationError: If validation fails and auto_validate is True
        """
        if not self._transaction_stack:
            raise NestedDataError("No active transaction")

        if self._validation_cache is not None:
            await self._validation_cache.clear()  # Clear entire cache on commit

        if self._auto_validate and self._schema:
            if not await self.validate():
                await self.rollback_transaction()
                raise ValidationError("transaction", "Schema validation failed")

        self._transaction_stack.pop()

    async def rollback_transaction(self) -> None:
        """
        Rollback the current transaction.

        Restores the state to what it was when the transaction began.

        Raises:
            NestedDataError: If no transaction is active
        """
        async with self.lock:
            if not self._transaction_stack:
                raise NestedDataError("No active transaction")
            self.data = self._transaction_stack.pop()
            if self._validation_cache is not None:
                await self._validation_cache.clear()

    async def batch_get(
            self,
            paths: Sequence[PathType],
            defaults: Optional[Sequence[Any]] = None,
            parallelize: bool = True
    ) -> Sequence[Any]:
        """
        Get multiple values in a single operation.

        Args:
            paths: List of paths to retrieve
            defaults: Optional list of default values (one per path)
            parallelize: Whether to parallelize the get operations

        Returns:
            List of retrieved values corresponding to the input paths

        Example:
            ```python
            values = await handler.batch_get([
                "users.123.name",
                "users.123.email",
                "users.123.settings"
            ], defaults=["Unknown", None, {}])
            ```
        """
        if defaults is None:
            defaults = [None] * len(paths)

        if parallelize:
            # Use cast to handle the typing of asyncio.gather()
            results = await asyncio.gather(
                *(self.get(path, default=default)
                  for path, default in zip(paths, defaults))
            )
            return cast(Sequence[Any], results)

        return [
            await self.get(path, default=default)
            for path, default in zip(paths, defaults)
        ]

    async def batch_set(
            self,
            updates: Dict[PathType, Any],
            create_missing: bool = True
    ) -> None:
        """
        Set multiple values in a single atomic operation.

        Performs all updates within a single lock acquisition for atomicity.
        If auto_validate is enabled, validates the final state after all updates.

        Args:
            updates: Dictionary mapping paths to their new values
            create_missing: Whether to create missing intermediate paths

        Raises:
            PathError: If any path is invalid
            ValidationError: If validation fails and auto_validate is True

        Example:
            ```python
            await handler.batch_set({
                "users.123.name": "John Doe",
                "users.123.email": "john@example.com",
                "users.123.settings.notifications": True
            })
            ```
        """
        async with self.lock:
            original_state = copy.deepcopy(self.data)

            try:
                for path, value in updates.items():
                    await self.set(path, value, create_missing)

                if self._auto_validate and self._schema:
                    if not await self.validate():
                        self.data = original_state
                        if self._validation_cache is not None:
                            await self._validation_cache.clear()
                        raise ValidationError("batch_set", "Validation failed")

            except Exception as e:
                self.data = original_state
                if self._validation_cache is not None:
                    await self._validation_cache.clear()
                raise e

    async def create_backup(self, label: Optional[str] = None) -> str:
        """
        Create a backup of the current state.

        Args:
            label: Optional label for the backup

        Returns:
            Backup identifier (label or auto-generated)

        Raises:
            NestedDataError: If backups are disabled

        Example:
            ```python
            backup_id = await handler.create_backup("pre_update")
            try:
                await handler.batch_set({...})
            except:
                await handler.restore_backup(backup_id)
            ```
        """
        if not self._backup_enabled:
            raise NestedDataError("Backups are disabled")

        async with self.lock:
            backup_id = label or f"backup_{len(self._backup_history)}"
            self._backup_history.append((backup_id, copy.deepcopy(self.data)))

            while len(self._backup_history) > self._max_backup_count:
                self._backup_history.pop(0)

            return backup_id

    async def restore_backup(self, identifier: Union[str, int] = -1) -> None:
        """
        Restore from a backup.

        Args:
            identifier: Backup identifier (label or index)

        Raises:
            NestedDataError: If backup not found or backups disabled

        Example:
            ```python
            # Restore by label
            await handler.restore_backup("pre_update")

            # Restore latest backup
            await handler.restore_backup()
            ```
        """
        if not self._backup_enabled or not self._backup_history:
            raise NestedDataError("No backups available")

        async with self.lock:
            if isinstance(identifier, str):
                for backup_id, backup_data in self._backup_history:
                    if backup_id == identifier:
                        self.data = copy.deepcopy(backup_data)
                        return
                raise NestedDataError(f"Backup '{identifier}' not found")
            else:
                try:
                    _, backup_data = self._backup_history[identifier]
                    self.data = copy.deepcopy(backup_data)
                except IndexError:
                    raise NestedDataError(f"Backup index {identifier} not found")

    async def validate(self, schema: Optional[SchemaType] = None, path: str = "") -> bool:
        """
        Validate the current data structure against a schema with caching.

        Args:
            schema: Optional schema to validate against (uses instance schema if None)
            path: Current path being validated (used for cache lookups)

        Returns:
            bool: True if validation succeeds

        Raises:
            ValueError: If no schema is available
        """
        schema = schema or self._schema
        if not schema:
            raise ValueError("No schema provided")

        if self._validation_cache is None and self._auto_validate:
            self._validation_cache = ValidationCache()

        async def _validate_value(value: Any, schema_part: dict, current_path: str = "") -> bool:
            try:
                if self._validation_cache is not None:
                    cached_result = await self._validation_cache.get_validation_result(
                        value, schema_part, current_path
                    )
                    if cached_result is not None:
                        return cached_result

                result = True

                # Type validation
                if "type" in schema_part:
                    result = result and self._validation_cache._validate_type(value, schema_part["type"])  # NOQA
                    if not result:
                        return False

                # Property validation for objects
                if "properties" in schema_part and isinstance(value, dict):
                    for key, sub_schema in schema_part["properties"].items():
                        if key == "*":
                            for k, v in value.items():
                                prop_path = f"{current_path}.{k}" if current_path else k
                                if not await _validate_value(v, sub_schema, prop_path):
                                    return False
                        elif key in value:
                            prop_path = f"{current_path}.{key}" if current_path else key
                            if not await _validate_value(value[key], sub_schema, prop_path):
                                return False
                        elif sub_schema.get("required", False):
                            return False

                # Array validation
                if "items" in schema_part and isinstance(value, list):
                    for i, item in enumerate(value):
                        item_path = f"{current_path}[{i}]" if current_path else str(i)
                        if not await _validate_value(item, schema_part["items"], item_path):
                            return False

                if self._validation_cache is not None:
                    await self._validation_cache.set_validation_result(
                        value, schema_part, True, current_path
                    )
                return True

            except Exception as e:
                logger.error(f"Validation error at {current_path}: {str(e)}")
                return False

        try:
            data_to_validate = await self.get(path) if path else self.data
            return await _validate_value(data_to_validate, schema, path)
        except Exception as e:
            logger.error(f"Validation error: {str(e)}")
            return False

    async def merge(
            self,
            other_data: Dict,
            strategy: MergeStrategy = "replace",
            path: Optional[PathType] = None
    ) -> None:
        """
        Merge another data structure into the current one.

        Args:
            other_data: Data structure to merge
            strategy: Merge strategy ('replace', 'update', or 'deep')
            path: Optional path where to perform the merge

        Raises:
            ValueError: If merge strategy is invalid
            PathError: If target path is invalid
            ValidationError: If validation fails and auto_validate is True

        Example:
            ```python
            # Deep merge at specific path
            await handler.merge(
                {"roles": ["admin"], "settings": {"theme": "dark"}},
                strategy="deep",
                path="users.123"
            )
            ```
        """

        async def deep_merge(target: Dict, source: Dict) -> Dict:
            for key, value in source.items():
                if (
                        key in target
                        and isinstance(target[key], dict)
                        and isinstance(value, dict)
                ):
                    await deep_merge(target[key], value)
                elif (
                        key in target
                        and isinstance(target[key], list)
                        and isinstance(value, list)
                ):
                    target[key].extend(value)
                else:
                    target[key] = copy.deepcopy(value)
            return target

        if strategy not in {"replace", "update", "deep"}:
            raise ValueError(f"Invalid merge strategy: {strategy}")

        async with self.lock:
            if path is None:
                target = self.data
            else:
                parent, key = await self._traverse(self.data, self._parse_path(path))
                if isinstance(parent, (dict, list)):
                    target = parent[key]
                else:
                    raise PathError(str(path), "Invalid merge target")

            if not isinstance(target, dict):
                raise DataTypeError("dict", type(target).__name__)

            if strategy == "replace":
                if path is None:
                    self.data = copy.deepcopy(other_data)
                else:
                    await self.set(path, copy.deepcopy(other_data))
            elif strategy == "update":
                target.update(other_data)
            else:  # strategy == "deep"
                await deep_merge(target, other_data)

            if self._validation_cache is not None:
                await self._validation_cache.invalidate_path(str(path) if path else "")

            if self._auto_validate and self._schema:
                if not await self.validate():
                    raise ValidationError("merge", "Validation failed")
