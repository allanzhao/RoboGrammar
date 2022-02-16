FROM ubuntu:18.04

# Install APT packages
RUN apt-get update && apt-get install -y \
cmake libglew-dev xorg-dev python3 python3-pip python3-dev \
python3-venv python-wheel libjpeg-dev zlib1g-dev
 

# Set PYTHONPATH
ENV PYTHONPATH="/robot_design/examples/design_search:/robot_design/examples/graph_learning:/robot_design/build/examples/python_bindings:$PYTHONPATH"


 
 
RUN python3 -m venv venv
RUN #!/bin/bash source venv/bin/activate
RUN pip3 install numpy 

RUN python3 -m pip install --upgrade --force-reinstall numpy-quaternion
RUN pip3 install torch==1.7.1+cpu -f https://download.pytorch.org/whl/torch_stable.html
RUN pip3 install torch-scatter==2.0.3 -f https://pytorch-geometric.com/whl/torch-1.7.0+cpu.html
RUN pip3 install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.7.0+cpu.html
RUN pip3 install --upgrade pip
RUN pip3 install llvmlite
RUN pip3 install torch-geometric==1.4.3 



# Copy our code and build
WORKDIR /robot_design
COPY . .
WORKDIR /robot_design/build
RUN cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo ..
RUN make -j8