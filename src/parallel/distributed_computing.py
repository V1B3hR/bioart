#!/usr/bin/env python3
"""
Distributed DNA Computing
Simple distributed computing framework for DNA programs
"""

import json
import socket
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional


class NodeType(Enum):
    """Distributed node types"""

    COORDINATOR = "coordinator"
    WORKER = "worker"
    HYBRID = "hybrid"


class TaskStatus(Enum):
    """Distributed task status"""

    PENDING = "pending"
    ASSIGNED = "assigned"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class DistributedTask:
    """Distributed DNA task"""

    task_id: str
    program: bytes
    input_data: Optional[bytes] = None
    priority: int = 5
    estimated_runtime: float = 1.0
    memory_requirements: int = 256
    assigned_node: Optional[str] = None
    status: TaskStatus = TaskStatus.PENDING
    created_time: float = 0.0
    started_time: Optional[float] = None
    completed_time: Optional[float] = None
    result: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.created_time == 0.0:
            self.created_time = time.time()


@dataclass
class ComputeNode:
    """Distributed compute node"""

    node_id: str
    node_type: NodeType
    address: str
    port: int
    capabilities: Dict[str, Any]
    current_load: float = 0.0
    max_concurrent_tasks: int = 4
    active_tasks: int = 0
    total_completed: int = 0
    total_failed: int = 0
    last_heartbeat: float = 0.0

    def __post_init__(self):
        if self.last_heartbeat == 0.0:
            self.last_heartbeat = time.time()


class DistributedDNAComputer:
    """
    Distributed DNA Computing System
    Coordinates DNA program execution across multiple nodes
    """

    def __init__(
        self,
        node_type: NodeType = NodeType.COORDINATOR,
        listen_port: int = 8080,
        node_id: Optional[str] = None,
    ):
        """Initialize distributed computer"""
        import uuid

        self.node_id = node_id or f"node_{uuid.uuid4().hex[:8]}"
        self.node_type = node_type
        self.listen_port = listen_port

        # Network components
        self.server_socket = None
        self.client_connections = {}
        self.network_thread = None
        self.running = False

        # Distributed state
        self.compute_nodes = {}
        self.distributed_tasks = {}
        self.task_assignments = {}
        self.global_task_counter = 0

        # Load balancing
        self.load_balancer = SimpleLoadBalancer()

        # Statistics
        self.stats = {
            "total_tasks_submitted": 0,
            "total_tasks_completed": 0,
            "total_tasks_failed": 0,
            "total_nodes_connected": 0,
            "average_task_time": 0.0,
            "network_messages_sent": 0,
            "network_messages_received": 0,
        }

        # VM factory for local execution
        self.vm_factory = None

        # Local worker capabilities
        if node_type in [NodeType.WORKER, NodeType.HYBRID]:
            self.local_worker = LocalWorker(self.node_id)

    def set_vm_factory(self, factory_func: Callable):
        """Set VM factory function"""
        self.vm_factory = factory_func
        if hasattr(self, "local_worker"):
            self.local_worker.set_vm_factory(factory_func)

    def start_network_service(self):
        """Start network service for distributed communication"""
        if self.running:
            return

        self.running = True
        self.network_thread = threading.Thread(target=self._network_service_loop)
        self.network_thread.daemon = True
        self.network_thread.start()

        print(f"Distributed DNA Computer {self.node_id} started on port {self.listen_port}")

    def stop_network_service(self):
        """Stop network service"""
        self.running = False

        if self.server_socket:
            self.server_socket.close()

        for conn in self.client_connections.values():
            conn.close()

        if self.network_thread:
            self.network_thread.join(timeout=5.0)

    def register_node(self, node_info: ComputeNode) -> bool:
        """Register a compute node"""
        if node_info.node_id in self.compute_nodes:
            return False

        self.compute_nodes[node_info.node_id] = node_info
        self.stats["total_nodes_connected"] += 1

        print(f"Node {node_info.node_id} registered ({node_info.node_type.value})")
        return True

    def submit_distributed_task(
        self,
        program: bytes,
        input_data: Optional[bytes] = None,
        priority: int = 5,
        task_id: Optional[str] = None,
    ) -> str:
        """Submit task for distributed execution"""
        if task_id is None:
            self.global_task_counter += 1
            task_id = f"dtask_{self.global_task_counter:06d}_{int(time.time())}"

        task = DistributedTask(
            task_id=task_id,
            program=program,
            input_data=input_data,
            priority=priority,
            estimated_runtime=len(program) * 0.001,  # Rough estimate
            memory_requirements=max(256, len(program) + (len(input_data) if input_data else 0)),
        )

        self.distributed_tasks[task_id] = task
        self.stats["total_tasks_submitted"] += 1

        # Assign task to node
        if self.node_type in [NodeType.COORDINATOR, NodeType.HYBRID]:
            self._assign_task_to_node(task)

        return task_id

    def _assign_task_to_node(self, task: DistributedTask):
        """Assign task to best available node"""
        available_nodes = [
            node
            for node in self.compute_nodes.values()
            if node.active_tasks < node.max_concurrent_tasks
        ]

        if not available_nodes:
            # Try local execution if this is a hybrid node
            if self.node_type == NodeType.HYBRID and hasattr(self, "local_worker"):
                self._execute_task_locally(task)
                return

            # Otherwise, queue task (would implement proper queuing)
            print(f"No available nodes for task {task.task_id}, queuing...")
            return

        # Select best node using load balancer
        best_node = self.load_balancer.select_node(available_nodes, task)

        if best_node:
            self._send_task_to_node(task, best_node)
        else:
            print(f"Failed to assign task {task.task_id}")

    def _send_task_to_node(self, task: DistributedTask, node: ComputeNode):
        """Send task to specific node"""
        try:
            # Create task message
            message = {
                "type": "execute_task",
                "task_id": task.task_id,
                "program": task.program.hex(),  # Hex encode for JSON
                "input_data": task.input_data.hex() if task.input_data else None,
                "priority": task.priority,
            }

            # Send to node
            if self._send_message_to_node(node.node_id, message):
                task.assigned_node = node.node_id
                task.status = TaskStatus.ASSIGNED
                node.active_tasks += 1
                self.task_assignments[task.task_id] = node.node_id

                print(f"Task {task.task_id} assigned to node {node.node_id}")
            else:
                print(f"Failed to send task {task.task_id} to node {node.node_id}")

        except Exception as e:
            print(f"Error sending task to node: {e}")

    def _execute_task_locally(self, task: DistributedTask):
        """Execute task on local worker"""
        if not hasattr(self, "local_worker"):
            return

        def local_execution():
            try:
                task.status = TaskStatus.RUNNING
                task.started_time = time.time()

                result = self.local_worker.execute_task(task)

                task.status = TaskStatus.COMPLETED if result["success"] else TaskStatus.FAILED
                task.completed_time = time.time()
                task.result = result

                if result["success"]:
                    self.stats["total_tasks_completed"] += 1
                else:
                    self.stats["total_tasks_failed"] += 1

                execution_time = task.completed_time - task.started_time
                self._update_average_task_time(execution_time)

                print(f"Local task {task.task_id} {'completed' if result['success'] else 'failed'}")

            except Exception as e:
                task.status = TaskStatus.FAILED
                task.completed_time = time.time()
                task.result = {"success": False, "error": str(e)}
                self.stats["total_tasks_failed"] += 1

        # Execute in separate thread
        execution_thread = threading.Thread(target=local_execution)
        execution_thread.daemon = True
        execution_thread.start()

    def _send_message_to_node(self, node_id: str, message: Dict[str, Any]) -> bool:
        """Send message to specific node"""
        # Simplified message sending (would implement proper networking)
        try:
            if node_id in self.client_connections:
                conn = self.client_connections[node_id]
                serialized = json.dumps(message).encode("utf-8")
                conn.send(len(serialized).to_bytes(4, "big"))
                conn.send(serialized)
                self.stats["network_messages_sent"] += 1
                return True
        except Exception as e:
            print(f"Failed to send message to {node_id}: {e}")

        return False

    def _network_service_loop(self):
        """Main network service loop"""
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            # Note: Binding to 0.0.0.0 allows connections from any interface
            # In production, consider using a specific host address for security
            self.server_socket.bind(("0.0.0.0", self.listen_port))
            self.server_socket.listen(10)

            while self.running:
                try:
                    client_socket, address = self.server_socket.accept()
                    client_thread = threading.Thread(
                        target=self._handle_client, args=(client_socket, address)
                    )
                    client_thread.daemon = True
                    client_thread.start()

                except OSError:
                    if self.running:
                        print("Socket error in network service")
                    break

        except Exception as e:
            print(f"Network service error: {e}")
        finally:
            if self.server_socket:
                self.server_socket.close()

    def _handle_client(self, client_socket: socket.socket, address):
        """Handle client connection"""
        try:
            while self.running:
                # Read message length
                length_data = client_socket.recv(4)
                if not length_data:
                    break

                message_length = int.from_bytes(length_data, "big")

                # Read message
                message_data = b""
                while len(message_data) < message_length:
                    chunk = client_socket.recv(message_length - len(message_data))
                    if not chunk:
                        break
                    message_data += chunk

                if len(message_data) == message_length:
                    try:
                        message = json.loads(message_data.decode("utf-8"))
                        self._process_message(message, client_socket)
                        self.stats["network_messages_received"] += 1
                    except json.JSONDecodeError:
                        print("Invalid JSON message received")

        except Exception as e:
            print(f"Client handler error: {e}")
        finally:
            client_socket.close()

    def _process_message(self, message: Dict[str, Any], client_socket: socket.socket):
        """Process received message"""
        msg_type = message.get("type")

        if msg_type == "register_node":
            self._handle_node_registration(message, client_socket)
        elif msg_type == "task_result":
            self._handle_task_result(message)
        elif msg_type == "heartbeat":
            self._handle_heartbeat(message)
        elif msg_type == "node_status":
            self._handle_node_status(message)
        else:
            print(f"Unknown message type: {msg_type}")

    def _handle_node_registration(self, message: Dict[str, Any], client_socket: socket.socket):
        """Handle node registration"""
        try:
            node_info = ComputeNode(
                node_id=message["node_id"],
                node_type=NodeType(message["node_type"]),
                address=message["address"],
                port=message["port"],
                capabilities=message.get("capabilities", {}),
                max_concurrent_tasks=message.get("max_concurrent_tasks", 4),
            )

            if self.register_node(node_info):
                self.client_connections[node_info.node_id] = client_socket

                # Send registration confirmation
                response = {"type": "registration_confirmed", "coordinator_id": self.node_id}
                self._send_response(client_socket, response)
            else:
                response = {"type": "registration_failed", "reason": "Node already registered"}
                self._send_response(client_socket, response)

        except Exception as e:
            print(f"Node registration error: {e}")

    def _handle_task_result(self, message: Dict[str, Any]):
        """Handle task completion result"""
        task_id = message.get("task_id")

        if task_id in self.distributed_tasks:
            task = self.distributed_tasks[task_id]
            task.status = TaskStatus.COMPLETED if message.get("success") else TaskStatus.FAILED
            task.completed_time = time.time()
            task.result = message.get("result", {})

            # Update node statistics
            if task.assigned_node in self.compute_nodes:
                node = self.compute_nodes[task.assigned_node]
                node.active_tasks = max(0, node.active_tasks - 1)

                if message.get("success"):
                    node.total_completed += 1
                    self.stats["total_tasks_completed"] += 1
                else:
                    node.total_failed += 1
                    self.stats["total_tasks_failed"] += 1

            # Update timing statistics
            if task.started_time:
                execution_time = task.completed_time - task.started_time
                self._update_average_task_time(execution_time)

            print(f"Task {task_id} {'completed' if message.get('success') else 'failed'}")

    def _handle_heartbeat(self, message: Dict[str, Any]):
        """Handle node heartbeat"""
        node_id = message.get("node_id")

        if node_id in self.compute_nodes:
            node = self.compute_nodes[node_id]
            node.last_heartbeat = time.time()
            node.current_load = message.get("current_load", 0.0)
            node.active_tasks = message.get("active_tasks", 0)

    def _handle_node_status(self, message: Dict[str, Any]):
        """Handle node status update"""
        node_id = message.get("node_id")

        if node_id in self.compute_nodes:
            node = self.compute_nodes[node_id]

            # Update node status
            if "current_load" in message:
                node.current_load = message["current_load"]
            if "active_tasks" in message:
                node.active_tasks = message["active_tasks"]

    def _send_response(self, client_socket: socket.socket, response: Dict[str, Any]):
        """Send response to client"""
        try:
            serialized = json.dumps(response).encode("utf-8")
            client_socket.send(len(serialized).to_bytes(4, "big"))
            client_socket.send(serialized)
            self.stats["network_messages_sent"] += 1
        except Exception as e:
            print(f"Failed to send response: {e}")

    def _update_average_task_time(self, execution_time: float):
        """Update average task execution time"""
        completed = self.stats["total_tasks_completed"]
        if completed > 0:
            current_avg = self.stats["average_task_time"]
            self.stats["average_task_time"] = (
                (current_avg * (completed - 1)) + execution_time
            ) / completed

    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get distributed task status"""
        if task_id not in self.distributed_tasks:
            return None

        task = self.distributed_tasks[task_id]

        return {
            "task_id": task_id,
            "status": task.status.value,
            "assigned_node": task.assigned_node,
            "created_time": task.created_time,
            "started_time": task.started_time,
            "completed_time": task.completed_time,
            "execution_time": (
                (task.completed_time - task.started_time)
                if task.started_time and task.completed_time
                else None
            ),
            "result": task.result,
        }

    def get_system_status(self) -> Dict[str, Any]:
        """Get distributed system status"""
        active_nodes = sum(
            1 for node in self.compute_nodes.values() if time.time() - node.last_heartbeat < 60.0
        )

        total_active_tasks = sum(node.active_tasks for node in self.compute_nodes.values())

        return {
            "node_id": self.node_id,
            "node_type": self.node_type.value,
            "total_nodes": len(self.compute_nodes),
            "active_nodes": active_nodes,
            "total_active_tasks": total_active_tasks,
            "system_load": sum(node.current_load for node in self.compute_nodes.values())
            / max(1, len(self.compute_nodes)),
            "statistics": self.stats.copy(),
            "nodes": [
                {
                    "node_id": node.node_id,
                    "node_type": node.node_type.value,
                    "address": f"{node.address}:{node.port}",
                    "active_tasks": node.active_tasks,
                    "current_load": node.current_load,
                    "last_heartbeat_age": time.time() - node.last_heartbeat,
                }
                for node in self.compute_nodes.values()
            ],
        }

    def wait_for_task(
        self, task_id: str, timeout: Optional[float] = None
    ) -> Optional[Dict[str, Any]]:
        """Wait for task completion"""
        start_time = time.time()

        while task_id in self.distributed_tasks:
            task = self.distributed_tasks[task_id]

            if task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
                return {
                    "success": task.status == TaskStatus.COMPLETED,
                    "result": task.result,
                    "execution_time": (
                        (task.completed_time - task.started_time)
                        if task.started_time and task.completed_time
                        else 0
                    ),
                }

            if timeout and (time.time() - start_time) >= timeout:
                return None

            time.sleep(0.1)

        return None


class SimpleLoadBalancer:
    """Simple load balancer for task assignment"""

    def select_node(
        self, available_nodes: List[ComputeNode], task: DistributedTask
    ) -> Optional[ComputeNode]:
        """Select best node for task"""
        if not available_nodes:
            return None

        # Score nodes based on current load and capabilities
        scored_nodes = []

        for node in available_nodes:
            # Calculate load factor
            load_factor = node.active_tasks / node.max_concurrent_tasks

            # Calculate capability score
            memory_factor = 1.0
            if "max_memory" in node.capabilities:
                memory_factor = min(1.0, node.capabilities["max_memory"] / task.memory_requirements)

            # Combine factors (lower is better)
            score = load_factor + (1.0 - memory_factor)
            scored_nodes.append((score, node))

        # Select node with lowest score
        scored_nodes.sort(key=lambda x: x[0])
        return scored_nodes[0][1]


class LocalWorker:
    """Local worker for executing tasks"""

    def __init__(self, worker_id: str):
        """Initialize local worker"""
        self.worker_id = worker_id
        self.vm_factory = None
        self.active_tasks = 0
        self.completed_tasks = 0
        self.failed_tasks = 0

    def set_vm_factory(self, factory_func: Callable):
        """Set VM factory"""
        self.vm_factory = factory_func

    def execute_task(self, task: DistributedTask) -> Dict[str, Any]:
        """Execute task locally"""
        if self.vm_factory is None:
            return {"success": False, "error": "VM factory not set"}

        try:
            self.active_tasks += 1

            # Create VM and execute
            vm = self.vm_factory()

            # Load input data
            if task.input_data:
                vm.memory[: len(task.input_data)] = task.input_data

            vm.load_program(task.program)

            start_time = time.time()
            vm.run()
            end_time = time.time()

            self.completed_tasks += 1

            return {
                "success": True,
                "execution_time": end_time - start_time,
                "final_registers": vm.registers.copy(),
                "output": getattr(vm, "output_buffer", []),
                "instructions_executed": getattr(vm, "instructions_executed", 0),
                "worker_id": self.worker_id,
            }

        except Exception as e:
            self.failed_tasks += 1
            return {"success": False, "error": str(e), "worker_id": self.worker_id}

        finally:
            self.active_tasks = max(0, self.active_tasks - 1)
