#final
from main import main
from tkinter import *
from tkinter import ttk
import threading
import os
import csv
import os
import re
import imaplib
import email
import csv
from datetime import datetime
import time
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email.mime.text import MIMEText
from email import encoders
import mods.calc,mods.pdf,mods.mail
import numpy as np
import pandas as pd
import subprocess
import utils.binvox_rw


def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
 
    return os.path.join(base_path, relative_path)

# Create a variable to control the loop
running = True

# Create an instance of tkinter frame or window
win = Tk()

# Set the size of the window
win.geometry("700x350")



# Define a function to print the text in a loop
def start_main():

    global running

    if not os.path.exists('credentials.txt'):
        raise FileNotFoundError(f"{filename} not found. Program will stop.")
    
    with open(resource_path('credentials.txt'), 'r') as file:
        contents = file.read()

    # Use regex to extract the values of your_email_address and your_password
    email_pattern = r"your_email_address=(.+)"
    password_pattern = r"your_password=(.+)"

    your_email_address = re.search(email_pattern, contents).group(1)
    your_password = re.search(password_pattern, contents).group(1)

    #print(your_email_address,your_password)

    send_mail=True

    #filename='my_pdf.pdf' 
    #df=pd.read_csv('invoice.csv')
    shapetypes = ['O ring', 'Through hole', 'Blind hole', 'Triangular passage', 'Rectangular passage', 'Circular through slot', 'Triangular through slot', 'Rectangular through slot', 'Rectangular blind slot','Triangular pocket', 'Rectangular pocket', 'Circular end pocket', 'Triangular blind step', 'Circular blind step', 'Rectangular blind step', 'Rectangular through step' , '2-sides through step', 'Slanted through step', 'Chamfer', 'Round', 'Vertical circular end blind slot', 'Horizontal circular end blind slot', '6-sides passage', '6-sides pocket']
    machtypes=shapetypes
    units=['','','(in m/min)','(Rs/min)','(in min)','(Rs)']

    '''labels=[[2.59220600e+00, 0.00000000e+00, 0.00000000e+00, 3.26974571e+03,
        3.25092912e+03, 9.98237908e+03, 1.90000000e+01, 9.99996781e-01],
       [4.08844948e+03, 3.68468940e+03, 0.00000000e+00, 6.25176549e+03,
        5.86772323e+03, 1.11457802e+03, 2.00000000e+00, 9.99766529e-01]]# 22'''
    tool = np.full((24, 5), [1, 1, 1, 1, 1])

    tool[1::2, :] = np.array([12.7,22,551.68,0.4,220.67])
    # Assign odd rows with [12,20,530.79,0.12,63.69]
    tool[::2, :] = np.array([12,20,530.79,0.12,63.69])
    #tool = np.full((24, 5), [1, 1,1,1,1])
    tool[19]=np.array([12,20,530.79,0.12,63.69])
    tool[2]=np.array([12.7,22,551.68,0.4,220.67])
    tool[1]=np.array([12.7,22,551.68,0.4,220.67])
    rate=np.ones([24,1])*7
    speed=np.ones([24,1])*21
    scale=10000


    # set up the IMAP connection
    mail = imaplib.IMAP4_SSL('imap.gmail.com')
    mail.login(your_email_address, your_password)

    # set the time interval for checking for new emails (in seconds)
    check_interval = 5

    # check if the file exists
    if not os.path.isfile(resource_path('attachment_log.csv')):
        # create the file with headers
        with open(resource_path('attachment_log.csv'), 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(['UID','Timestamp','Sender','filenames'])

    if not os.path.exists(resource_path('Staging Area')):
        os.makedirs(resource_path('Staging Area'))

    print('6')

    while running:
    	main(mail,units,tool,rate,speed,scale,shapetypes,machtypes,send_mail,your_email_address,your_password,check_interval)
    

# Define a function to start the loop
def on_start():
    global running
    running = True
    # Start the loop in a new thread to avoid blocking the GUI
    threading.Thread(target=start_main).start()

# Define a function to stop the loop
def on_stop():
    global running
    running = False
    
canvas = Canvas(win, bg="skyblue3", width=600, height=60)
canvas.create_text(150, 10, text="Click the Start/Stop to execute the Code", font=('', 13))
canvas.pack()

# Add a Button to start/stop the loop
start = ttk.Button(win, text="Start", command=on_start)
start.pack(padx=10)

stop = ttk.Button(win, text="Stop", command=on_stop)
stop.pack(padx=10)

win.mainloop()
