import flwr as fl
import os
import sys
import subprocess

numberOfclients = int(float(str(sys.argv[1])))
if numberOfclients < 1:
    print('Number of clients is not specified!')
else:
    command_torun_clients = ['python3','client.py']
    #command_torun_server = ['python3','server.py']
    for i in range(numberOfclients):
        #command_torun_clients = command_torun_clients + '& python3 client.py '
        command_torun_clients.append('&')
        command_torun_clients.append('python3')
        command_torun_clients.append('client.py')

    #os.system(command_torun_server)
    #subprocess.run(command_torun_server)
    subprocess.run(command_torun_clients)
    #os.system(command_torun_clients)
