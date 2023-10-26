#include "mmio.h"
#include "rose.h"

#include <stdio.h>
#include <inttypes.h>
#include <string.h>
#include <riscv-pk/encoding.h>


#define ROSE_REQ_PID_OBS 0x16

#define ROSE_SET_TARGETS  0x20

// #define KP_ANGLE 10.0f
// #define KD_ANGLE 0.1f
// #define KP_POS 1.0f
// #define KD_POS 0.0f

#define KP_ANGLE 41.20f
#define KD_ANGLE 2.16f
#define KP_POS 5.26f
#define KD_POS 1.14f





uint32_t buf[8];

int read_pid_obs_nonblock(float * obs) {
	send_obs_req(ROSE_REQ_PID_OBS);
  return read_obs_rsp_nonblock((void *) obs);
}

float calc_action(float * obs) {
  float theta = obs[1];
  float theta_dot = obs[3];

  float pos = obs[0];
  float vel = obs[2];

  float action = KP_ANGLE * theta + KD_ANGLE * theta_dot + KP_POS * pos + KD_POS * vel;
  return action;
}

int main(void)
{
  uint32_t cnt = 0;
  float obs[4] = {0, 0, 0, 0};
  float action;

  // printf("Starting pd test code: KP: %f, KD: %f, KP_POS: %f, KD_POS: %f ...\n", KP_ANGLE, KD_ANGLE, KP_POS, KD_POS);
  printf("Starting pd test code.\n");
  printf("Sending simulator reset...\n");
  send_sim_reset();

  while(1) {

    printf("Requesting observation %d\n", cnt);
    read_pid_obs_nonblock(obs);
    //printf("Received observation %d: [%f, %f, %f, %f]\n", cnt, obs[0], obs[1], obs[2], obs[3]);

    action = calc_action(obs);
    //printf("Computed action: %f\n", action);
    printf("Computed action.\n");
    send_action(&action, ROSE_SET_TARGETS, 4);
    cnt++;
  }
}

