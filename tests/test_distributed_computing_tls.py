import socket
import types
import unittest
from unittest import mock

import src.parallel.distributed_computing as distributed_computing


class _FakeWrappedSocket:
    def __init__(self, computer):
        self._computer = computer
        self.closed = False

    def settimeout(self, timeout):
        self.timeout = timeout
        self._computer.running = False

    def accept(self):
        raise socket.timeout()

    def close(self):
        self.closed = True


class _FakeContext:
    def __init__(self, wrapped_socket, has_minimum_version=True):
        self._wrapped_socket = wrapped_socket
        self.options = 0
        self.load_cert_chain = mock.Mock()
        self.wrap_socket = mock.Mock(return_value=wrapped_socket)
        if has_minimum_version:
            self.minimum_version = None


class DistributedComputingTLSTests(unittest.TestCase):
    def _run_network_loop(self, *, server_protocol_available, tls_version_available=True):
        computer = distributed_computing.DistributedDNAComputer(
            listen_port=9123,
            ssl_certfile="cert.pem",
            ssl_keyfile="key.pem",
        )
        wrapped_socket = _FakeWrappedSocket(computer)
        context = _FakeContext(
            wrapped_socket,
            has_minimum_version=tls_version_available,
        )
        ssl_context_factory = mock.Mock(return_value=context)
        fake_ssl = types.SimpleNamespace(
            SSLContext=ssl_context_factory,
            PROTOCOL_TLS="tls",
            OP_NO_TLSv1=0x1,
            OP_NO_TLSv1_1=0x2,
        )
        if server_protocol_available:
            fake_ssl.PROTOCOL_TLS_SERVER = "tls-server"
        if tls_version_available:
            fake_ssl.TLSVersion = types.SimpleNamespace(TLSv1_2="tls1.2")

        base_socket = mock.Mock()

        with mock.patch.object(distributed_computing, "ssl", fake_ssl), mock.patch.object(
            distributed_computing.socket,
            "socket",
            return_value=base_socket,
        ):
            computer.running = True
            computer._network_service_loop()

        return computer, base_socket, context, ssl_context_factory, wrapped_socket

    def test_network_service_uses_server_tls_context_when_available(self):
        computer, base_socket, context, ssl_context_factory, wrapped_socket = self._run_network_loop(
            server_protocol_available=True,
        )

        base_socket.bind.assert_called_once_with(("127.0.0.1", 9123))
        base_socket.listen.assert_called_once_with(10)
        ssl_context_factory.assert_called_once_with("tls-server")
        self.assertEqual(context.minimum_version, "tls1.2")
        self.assertEqual(context.options, 0x3)
        context.load_cert_chain.assert_called_once_with(certfile="cert.pem", keyfile="key.pem")
        context.wrap_socket.assert_called_once_with(base_socket, server_side=True)
        self.assertIs(computer.server_socket, wrapped_socket)
        self.assertTrue(wrapped_socket.closed)

    def test_network_service_falls_back_without_server_protocol(self):
        _, _, context, ssl_context_factory, wrapped_socket = self._run_network_loop(
            server_protocol_available=False,
            tls_version_available=False,
        )

        ssl_context_factory.assert_called_once_with("tls")
        self.assertEqual(context.options, 0x3)
        self.assertFalse(hasattr(context, "minimum_version"))
        self.assertTrue(wrapped_socket.closed)


if __name__ == "__main__":
    unittest.main()
