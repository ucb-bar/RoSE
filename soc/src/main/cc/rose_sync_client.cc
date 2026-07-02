/*
 * Copyright (c) 2026 UC Berkeley
 * SPDX-License-Identifier: Apache-2.0
 *
 * rose_sync_client implementation (see header). Socket + wire framing ported from
 * rosebridge.cc's synchronizer protocol.
 */

#include "rose_sync_client.h"

#include <cstdio>
#include <cstring>
#include <unistd.h>
#include <netdb.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <fcntl.h>
#include <errno.h>

rose_sync_client_t::rose_sync_client_t(const char *host, int port)
	: hostname(host), portno(port), sockfd(-1), connected(false) {}

bool rose_sync_client_t::open_socket()
{
	if (sockfd >= 0) {
		return true;
	}
	sockfd = socket(AF_INET, SOCK_STREAM | SOCK_NONBLOCK, 0);
	if (sockfd < 0) {
		perror("[rose_sync] socket");
		return false;
	}
	return true;
}

bool rose_sync_client_t::ensure_connected()
{
	if (connected) {
		return true;
	}
	if (!open_socket()) {
		return false;
	}
	struct hostent *server = gethostbyname(hostname);
	if (server == nullptr) {
		return false;
	}
	struct sockaddr_in addr;
	memset(&addr, 0, sizeof(addr));
	addr.sin_family = AF_INET;
	memcpy(&addr.sin_addr.s_addr, server->h_addr, server->h_length);
	addr.sin_port = htons(portno);

	int r = connect(sockfd, (const struct sockaddr *)&addr, sizeof(addr));
	if (r == 0 || errno == EISCONN) {
		connected = true;
		fprintf(stderr, "[rose_sync] connected to %s:%d\n", hostname, portno);
		return true;
	}
	/* EINPROGRESS / EALREADY: still connecting; try again next call. */
	return false;
}

rose_sync_client_t::~rose_sync_client_t()
{
	if (sockfd >= 0) {
		close(sockfd);
	}
}

bool rose_sync_client_t::connect_blocking()
{
	while (!ensure_connected()) {
		usleep(1000);
	}
	return true;
}

/* Read exactly nwords (non-blocking, all-or-nothing over a short spin). */
bool rose_sync_client_t::read_words(uint32_t *dst, uint32_t nwords)
{
	size_t want = (size_t)nwords * 4;
	size_t got = 0;
	uint8_t *p = (uint8_t *)dst;
	int spins = 0;
	while (got < want) {
		ssize_t n = read(sockfd, p + got, want - got);
		if (n > 0) {
			got += (size_t)n;
		} else if (n == 0) {
			return false; /* peer closed */
		} else {
			if (errno == EAGAIN || errno == EWOULDBLOCK) {
				if (got == 0) {
					return false; /* nothing yet */
				}
				if (++spins > 1000000) {
					return false;
				}
				continue; /* mid-packet: keep waiting for the rest */
			}
			return false;
		}
	}
	return true;
}

bool rose_sync_client_t::read_words_blocking(uint32_t *dst, uint32_t nwords)
{
	int backoff = 1;
	while (!read_words(dst, nwords)) {
		usleep(backoff);
		if (backoff < 1024) {
			backoff *= 2;
		}
	}
	return true;
}

void rose_sync_client_t::write_words(const uint32_t *src, uint32_t nwords)
{
	size_t want = (size_t)nwords * 4;
	size_t sent = 0;
	const uint8_t *p = (const uint8_t *)src;
	while (sent < want) {
		ssize_t n = write(sockfd, p + sent, want - sent);
		if (n > 0) {
			sent += (size_t)n;
		} else if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
			usleep(1);
		} else if (n < 0) {
			perror("[rose_sync] write");
			return;
		}
	}
}

void rose_sync_client_t::send_tx(uint32_t cmd, const uint32_t *data, uint32_t nwords)
{
	uint32_t hdr[2] = {cmd, nwords * 4};
	write_words(hdr, 2);
	if (nwords > 0 && data != nullptr) {
		write_words(data, nwords);
	}
}

bool rose_sync_client_t::try_recv(rose_pkt_t &out)
{
	uint32_t cmd;
	if (!read_words(&cmd, 1)) {
		return false; /* nothing available */
	}
	out.cmd = cmd;
	out.big_step = 0;
	out.budget = 0;
	out.data.clear();
	out.is_control = (cmd >= 0x80);

	uint32_t num_bytes = 0;
	if (out.is_control) {
		read_words_blocking(&num_bytes, 1);
	} else {
		read_words_blocking(&out.big_step, 1);
		read_words_blocking(&out.budget, 1);
		read_words_blocking(&num_bytes, 1);
	}
	uint32_t nwords = num_bytes / 4;
	if (nwords > 0) {
		out.data.resize(nwords);
		read_words_blocking(out.data.data(), nwords);
	}
	return true;
}

bool rose_sync_client_t::recv_blocking(rose_pkt_t &out)
{
	int backoff = 1;
	while (!try_recv(out)) {
		usleep(backoff);
		if (backoff < 1024) {
			backoff *= 2;
		}
	}
	return true;
}
