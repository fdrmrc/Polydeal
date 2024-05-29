IMG=dealii/dealii:master-focal
docker run  --user $(id -u):$(id -g) \
    --rm -t \
    -v `pwd`:/data $IMG /bin/sh -c "cd /data/; ./build/examples/export_matrices"
