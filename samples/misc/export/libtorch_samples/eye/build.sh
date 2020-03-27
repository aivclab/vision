LIBPYTORCH_PATH=$HOME/Applications/libtorch
mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH=$LIBPYTORCH_PATH ..
make
cd ..
