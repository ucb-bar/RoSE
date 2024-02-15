//RoSE Control Packet Headers
#define CS_RESET 0xff
#define CS_REQ_WAYPOINT 0xf1
#define CS_RSP_WAYPOINT 0xf2
#define CS_SEND_IMU 0xf3
#define CS_REQ_ARM 0xf4
#define CS_REQ_DISARM 0xf5
#define CS_REQ_TAKEOFF 0xf6
#define CS_GRANT_TOKEN 0x80
#define CS_REQ_CYCLES 0x81
#define CS_RSP_CYCLES 0x82
#define CS_DEFINE_STEP 0x83
#define CS_RSP_STALL 0x84
#define CS_CFG_BW 0x85
#define CS_CFG_ROUTE 0x86
//RoSE Payload Packet Headers
#define CS_CAMERA 0x10
#define CS_IMU 0x20
#define CS_ACCELEROMETER 0x22
#define CS_GYROSCOPE 0x24