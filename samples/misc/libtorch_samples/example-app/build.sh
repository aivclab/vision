LIBPYTORCH_PATH=../../libtorch
mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH=$LIBPYTORCH_PATH ..
make
cd ..
