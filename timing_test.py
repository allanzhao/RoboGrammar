import threading
import time
import numpy


from multiprocessing import Process, Value, Array      

#this are Radians that Main.py sends to Controler.py (controler.move(goal_pos[]))
goal_pos = [0.35834162, 0.05228215, -0.52554983, -0.05782359, -0.86107714,  1.26806711,  1.45226386,  1.8927399,  -0.29411051,  0.24729285] 
#Hz depens on how fast you recieve move command
speed_factor = 25 
#start position in middle 0-4095
present_POS   = [2045, 2045, 2045, 2045, 2045, 2045, 2045, 2045, 2045, 2045, 2045]

start = time.time()
for s in range(0, speed_factor, 1):
    for i in range(0, len(goal_pos), 1):

        calculated_position_goal_pos = int(2045 + (numpy.rad2deg(goal_pos[i]) * (4095 / 360)))

        speed_position = present_POS[i] + int((calculated_position_goal_pos - present_POS[i]) / speed_factor)
            
        print("motor: ", i," ", "new position: ", speed_position, "timestamp: ", s)

        #self.dxl_comm_result, self.dxl_error = self.packetHandler.write4ByteTxRx(self.portHandler, self.motor_ID[i], self.addr_goal_position, speed_position)

        present_POS[i] = speed_position

    time.sleep(1 / speed_factor)

    print(s)
end = time.time()
total = end - start
print(total)
