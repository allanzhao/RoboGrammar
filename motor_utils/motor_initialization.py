import sys
sys.path.append("/home/biobot/BiobotGrammar/")

from controller import Controller
from time import sleep
import json


if __name__ == "__main__":
    controller = Controller()
    MAX_POS_INIT_ADDR = 48
    MIN_POS_INIT_ADDR = 52
    print(controller.motor_num)
    
    motor_config = json.load(open("../configs/292_motors.json"))
    
    for i in range(controller.motor_num):
        # set joint limits
        print("MOTOR: %d (id: %d)" % (i, controller.motor_ID[i]))
        min_pos = motor_config["min_limits"][str(controller.motor_ID[i])]
        _, dxl_error = controller.packetHandler.write4ByteTxRx(controller.portHandler, controller.motor_ID[i], MIN_POS_INIT_ADDR, 0)
        controller.log_error(i, dxl_error)
        sleep(0.1)
        max_pos = motor_config["max_limits"][str(controller.motor_ID[i])]
        _, dxl_error = controller.packetHandler.write4ByteTxRx(controller.portHandler, controller.motor_ID[i], MAX_POS_INIT_ADDR, 4095)
        controller.log_error(i, dxl_error)
        sleep(0.1)
        
        