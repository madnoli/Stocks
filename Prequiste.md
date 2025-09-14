Prequiste 

1. Source Environment Requirements
    Oracle VM Version: Oracle VM 3.4 or later
    VM State: All VMs must be shut down before migration begins
    Network Connectivity: SSH access between source and destination hosts
    Storage Access: VM disk images must be accessible to both source and destination systems
    Administrative access to Oracle VM hosts
    VM repository access and export capabilities


2. Target Environment (OLVM)
    Oracle Linux 8.5 or later for the OLVM engine
    KVM hosts configured and added to OLVM cluster
    Storage domains configured in OLVM

2. Software Requirements
    Operating System: Oracle Linux 8.8 or later Oracle Linux 8 release
    Base Installation: Minimal Install base environment only
    Passwordless SSH authentication between hosts
    SSH access from KVM host to Oracle VM host
    Network Ports:
        Port 22 (SSH)
        Port 16509 (TLS connections)
        Port 16514 (TCP connections)
        Ports 49152-49215 (QEMU migration data)

4. Software and Tools
    Required Packages
        virt-v2v tool for VM conversion
        libvirt daemon on Oracle VM host
        qemu-block-curl RPM on target KVM server
        libguestfs tools for disk image handling

Important: Internet access to OVM/OVS and OLVM (KVM server and OLVM Manager) to intall any required packges. 