//RoSE Control Packet Headers
#define CS_RESET 0xff
#define CS_GRANT_TOKEN 0x80
#define CS_REQ_CYCLES 0x81
#define CS_RSP_CYCLES 0x82
#define CS_DEFINE_STEP 0x83
#define CS_RSP_STALL 0x84
#define CS_CFG_BW 0x85
#define CS_CFG_ROUTE 0x86
//RoSE Payload Packet Headers
#define CS_CAMERA_STEREO 0x10
#define CS_CAMERA_LEFT 0x11
#define CS_CAMERA_RIGHT 0x12
#define CS_CAMERA_LOOPBACK 0x01