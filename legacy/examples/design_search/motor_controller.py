from time import sleep
import traceback

from multiprocessing import Process, Value, Array


class Controller(Process):
    
    def __init__(self):
        super(Controller, self).__init__()
        # this makes it shut down when the main process terminates.
        self.deamon = True
        
        # initialize your buffers here.
        # Value and Array structures should be used to ensure multithread-safe reading and writing.
        # see https://github.com/benquick123/BiobotGrammar/blob/main/examples/design_search/neurons.py#L33 for example use-case.
        
    def run(self):
        # this is called when the controller.start() is run.
        # should probably contain a while loop waiting for new motor commands
        # and taking care of sending commands to the motors.
        
        # delete next line when you implement your code.
        raise NotImplementedError
    
    def stop(self):
        # if you need to run any code to release the communication channels,
        # execute it here.
        pass
    
    def move(self, positions, in_time=1/15):
        """
        arguments:
            positions: absolute positions the motors should reach
            in_time: time that the movement should take. Optional.
        """
        # take care of the movement logic. 
        # at very least save the positions to your multithread-safe buffer variables.
        pass


if __name__ == "__main__":
    # if your run this file by executing `python motor_controller.py` code 
    # in this if statement will be executed.
    
    # define the controller
    controller = Controller()
    # start the process
    controller.start()
    
    try:
        # send commands to the process by executing
        # controller.move({some numbers})
        # either inside or outside the loop.
        while True:
            sleep(0.1)
    except KeyboardInterrupt:
        # when you press CTRL+C, the whole program gracefully stops.
        controller.stop()
    except:
        # if any other error occurs, it will be printed here.
        traceback.print_exc()
        controller.stop()