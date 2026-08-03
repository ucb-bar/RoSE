/*
 * Copyright (c) 2026 UC Berkeley
 * SPDX-License-Identifier: Apache-2.0
 *
 * rose_sync_client: transport/sim-agnostic client for the RoSE synchronizer
 * protocol (TCP to the gym synchronizer on :10001). Factored so it can back both
 * the FireSim host bridge driver and the Spike device plugin.
 *
 * Wire framing (matches rosebridge.cc / gym_synchronizer.py), all little-endian
 * uint32 words:
 *   synchronizer -> bridge, control (cmd >= 0x80): [cmd][num_bytes][data...]
 *   synchronizer -> bridge, data    (cmd <  0x80): [cmd][big_step][budget][num_bytes][data...]
 *   bridge -> synchronizer (SoC TX):               [cmd][num_bytes][data...]
 */

#ifndef ROSE_SYNC_CLIENT_H_
#define ROSE_SYNC_CLIENT_H_

#include <cstdint>
#include <vector>

/* RoSE control/payload command set (mirrors rose_packet.h). */
#define ROSE_CS_RESET        0xff
#define ROSE_CS_GRANT_TOKEN  0x80
#define ROSE_CS_REQ_CYCLES   0x81
#define ROSE_CS_RSP_CYCLES   0x82
#define ROSE_CS_DEFINE_STEP  0x83
#define ROSE_CS_RSP_STALL    0x84
#define ROSE_CS_CFG_BW       0x85
#define ROSE_CS_CFG_ROUTE    0x86

/* One decoded inbound packet from the synchronizer. */
struct rose_pkt_t {
	uint32_t cmd;
	uint32_t big_step;              /* data packets only */
	uint32_t budget;               /* data packets only */
	std::vector<uint32_t> data;    /* num_bytes/4 words */
	bool is_control;               /* cmd >= 0x80 */
};

class rose_sync_client_t {
public:
	rose_sync_client_t(const char *host = "localhost", int port = 10001);
	~rose_sync_client_t();

	/* Block until connected to the synchronizer. Returns false on fatal error. */
	bool connect_blocking();

	/*
	 * Lazy, NON-blocking connect: attempt to (finish) connecting without spinning.
	 * Returns true once connected, false if not yet (call again later). Safe to call
	 * every pump; never hangs. Use this from a Spike device so construction can't
	 * stall the simulator before the synchronizer is up.
	 */
	bool ensure_connected();

	bool is_connected() const { return connected; }

	/* True once the synchronizer has closed the connection (read()==0 / EOF).
	 * Lets the harness distinguish "peer gone" from "no data yet" (both make
	 * read_words return false) and shut down instead of busy-spinning forever. */
	bool peer_closed() const { return peer_closed_; }

	/* Send a SoC TX packet [cmd][num_bytes][data...] to the synchronizer. */
	void send_tx(uint32_t cmd, const uint32_t *data, uint32_t nwords);

	/*
	 * Try to receive ONE fully-framed packet (non-blocking). Returns true and
	 * fills @p out if a complete packet was available, false otherwise.
	 */
	bool try_recv(rose_pkt_t &out);

	/* Block (spin with backoff) until one packet is received. */
	bool recv_blocking(rose_pkt_t &out);

	int fd() const { return sockfd; }

private:
	bool read_words(uint32_t *dst, uint32_t nwords); /* non-blocking; all-or-nothing */
	bool read_words_blocking(uint32_t *dst, uint32_t nwords);
	void write_words(const uint32_t *src, uint32_t nwords);

	bool open_socket();

	const char *hostname;
	int portno;
	int sockfd;
	bool connected;
	bool peer_closed_ = false;
};

#endif /* ROSE_SYNC_CLIENT_H_ */
