FROM redhat/ubi10-minimal

RUN microdnf install -y python3 python3-pip git gcc gcc-c++ make cmake findutils which \
    && microdnf clean all
