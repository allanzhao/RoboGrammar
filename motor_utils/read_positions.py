import sys
sys.path.append("/home/biobot/BiobotGrammar/")

from controller import Controller
from time import sleep

if __name__ == "__main__":
    controller = Controller()
    motor_indexes = {9}
    
    try:
        while True:
            for i in range(controller.motor_num):
                if i not in motor_indexes:
                    continue
                dxl_present_position, _, dxl_error = controller.packetHandler.read4ByteTxRx(controller.portHandler, controller.motor_ID[i], controller.addr_present_position)
                controller.log_error(i, dxl_error)
                
                print("MOTOR: %d (id: %d), POSITION: %d" % (i, controller.motor_ID[i], dxl_present_position))
                sleep(0.1)
    except KeyboardInterrupt:
        controller.stop()
        