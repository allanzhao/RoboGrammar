import traceback
import time
import numpy


from multiprocessing import Process, Value, Array

from dynamixel_sdk import * # Uses Dynamixel SDK library


class Controller(Process):
    
    def __init__(self):
        super(Controller, self).__init__()
        # this makes it shut down when the main process terminates.
        self.deamon = True
        
        # initialize your buffers here.
        # Value and Array structures should be used to ensure multithread-safe reading and writing.
        # see https://github.com/benquick123/BiobotGrammar/blob/main/examples/design_search/neurons.py#L33 for example use-case.



        self.motor_ID               = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        self.motor_num              = len(self.motor_ID)
                  
        
        #Read and Write data from and to motors

        # EEPROM
        
        # self.homing_offset          = Array('i', [0] * self.motor_num)
        # self.min_position           = Array('i', [0] * self.motor_num)
        # self.max_position           = Array('i', [0] * self.motor_num)
        
        self.goal_position          = Array('i', [0] * self.motor_num)
        self.present_POS            = Array('i', [0] * self.motor_num)
        self.torque_enable          = Array('i', [0] * self.motor_num) #boolean
        self.led                    = Array('i', [0] * self.motor_num) #boolean



        #Addresses
        self.addr_torque_enable     = 64
        self.addr_homing_offset     = 20
        self.addr_min_position      = 52
        self.addr_max_position      = 48
        self.addr_goal_position     = 116
        self.addr_present_position  = 132
        self.addr_led               = 65
            #Dynamixel XL430 250T settings
        self.boudrate               = 3000000
        self.protocol_type          = 2
        self.DEVICENAME             = 'COM7'


        self.portHandler = PortHandler(self.DEVICENAME) # ex) Windows: "COM*", Linux: "/dev/ttyUSB*", Mac: "/dev/tty.usbserial-*"
        self.packetHandler = PacketHandler(self.protocol_type)

        

    #def run(self):
        # this is called when the controller.start() is run.
        # should probably contain a while loop waiting for new motor commands
        # and taking care of sending commands to the motors.
        
        # delete next line when you implement your code.
        # raise NotImplementedError

            #Dynamixel motor IDs 


    #Device Communication

        # Open port (if needed uncomment)
        if self.portHandler.openPort():
            print("Succeeded to open the port")
        else:
            print("Failed to open the port")
            print("%s" % self.packetHandler.getRxPacketError(self.dxl_error))

        # Open port (if needed uncomment)
        if self.portHandler.setBaudRate(self.boudrate):
            print("Succeeded to change the baudrate")
        else:
            print("Failed to change the baudrate")
            quit()


            # loop throug arrays of motor IDs read position . if succesfull, set goals position and turn on torque
        for i in range(0, self.motor_num - 1, 1):

            #if self.dxl_comm_result != COMM_SUCCESS:
           
            dxl_present_position, self.dxl_comm_result, self.dxl_error = self.packetHandler.read4ByteTxRx(self.portHandler, self.motor_ID[i], self.addr_present_position)

            if self.dxl_error != 0:
                print("%s" % self.packetHandler.getRxPacketError(dxl_error), "error")
                

                    # Set present position 
            self.present_POS[i]   = dxl_present_position

            self.dxl_comm_result, dxl_error = self.packetHandler.write1ByteTxRx(self.portHandler, self.motor_ID[i], self.addr_led, 1)
            self.dxl_comm_result, dxl_error = self.packetHandler.write4ByteTxRx(self.portHandler, self.motor_ID[i], self.addr_goal_position, self.present_POS[i])
            self.dxl_comm_result, dxl_error = self.packetHandler.write1ByteTxRx(self.portHandler, self.motor_ID[i], self.addr_torque_enable, 1)

            #time.sleep(0.5)
            #print(i, self.present_POS[i], " position")
    def stop(self):
        # if you need to run any code to release the communication channels,
        # execute it here.
        

        #just turn off the torque. 

        #also need to see if dynamixelSDK closes port after termination(probably does on its own)
        for i in range(0, (self.motor_num - 1), 1):

            self.dxl_comm_result, self.dxl_error = self.packetHandler.write1ByteTxRx(self.portHandler, self.motor_ID[i], self.addr_torque_enable, 0)
   
    def move(self, goal_pos):
        """
        arguments:
            positions: absolute positions the motors should reach
            in_time: time that the movement should take. Optional.
        """
        #degrees = (goal_pos[i] * 180) / math.pi 
        calculated_position_goal_pos = int(2045 + (numpy.rad2deg(goal_pos) * (4095 / 360))) #this need to round up better
        
        #Hz. should be 25hz. depents on how fast main.py sends controller.move(goal_pos[])
        speed_factor = 25

        # start = time.time()

        for s in range(0, speed_factor, 1):
            for i in range(0, len(goal_pos), 1):


                speed_position = self.present_POS[i] + int((calculated_position_goal_pos - self.present_POS[i]) / speed_factor)
                #print("motor: ", i," ", "new position: ", speed_position, "timestamp: ", s)

                self.dxl_comm_result, self.dxl_error = self.packetHandler.write4ByteTxRx(self.portHandler, self.motor_ID[i], self.addr_goal_position, speed_position)

                self.present_POS[i] = speed_position

        time.sleep(1 / speed_factor)

        # print(s)
        # end = time.time()
        # total = end - start
        # print(total)

        
        # take care of the movement logic. 
        # at very least save the positions to your multithread-safe buffer variables.
        

    def firstMove(self):

        pass




if __name__ == "__main__":
    # if your run this file by executing `python motor_controller.py` code 
    # in this if statement will be executed.
    
    # define the controller
    controller = Controller()
    # start the process

    
    #try:
        # send commands to the process by executing
        # controller.move({some numbers})
        # either inside or outside the loop.
        
        
    controller
        #This should simulate sending controller.move for one ID
    for i in range(0, 4000, 50):
        print(i)
        controller.move(1, i)
    time.sleep(1)
    

    '''
        except KeyboardInterrupt:
        # when you press CTRL+C, the whole program gracefully stops.
        #controller.stop()
        #except:
        if #any other error occurs, it will be printed here.
            traceback.print_exc()
        #controller.stop()
    '''
