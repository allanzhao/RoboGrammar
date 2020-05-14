FROM ubuntu:18.04

# Install APT packages
RUN apt-get update && apt-get install -y cmake libglew-dev xorg-dev python3 python3-pip python3-dev

# Install third party Python packages
RUN pip3 install numpy numpy-quaternion

# Set PYTHONPATH
ENV PYTHONPATH="/robot_design/examples/design_search:/robot_design/examples/graph_learning:/robot_design/build/examples/python_bindings:$PYTHONPATH"

# Copy our code and build
WORKDIR /robot_design
COPY . .
WORKDIR /robot_design/build
RUN cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo ..
RUN make -j8

WORKDIR /robot_design
CMD python3 examples/design_search/design_search.py
