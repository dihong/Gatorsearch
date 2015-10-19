# Online product search engine
> 1. Collect tens of millions of product items (both images and text) from the web.
> 2. Designed effective algorithms (combining text and image information) that improves the accuracy over the state-of-art image retrieval systems.
> 3. Implemented scalable search engine for real-time product query.

# Install and Run
make <br />
mpirun --mca btl_tcp_if_include eth0 -np 4 --hostfile ./mpi_hostfile bin/gatorsearch

# Developers
Dihong Gong, Siliang Xia, Siva Prasad
