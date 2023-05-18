import os
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
import pandas as pd
import numpy as np

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
 
    return os.path.join(base_path, relative_path)

def generate_mail(has_stl_attachment,send_mail,your_email_address,your_password,sender):
    if has_stl_attachment:
        if send_mail:
            # Set up the message
            msg = MIMEMultipart()
            msg['From'] = your_email_address
            msg['To'] =  sender
            msg['Subject'] = 'Quote Results'
            # Add the text message
            text = MIMEText('Dear customer,\n\n    Please find your requested quote attached.\n\nBest regards,\nThe Sales Team')
            msg.attach(text)

            # Add the attachments
            for filename in os.listdir(resource_path('Staging Area')):
                if filename.endswith('.pdf') or filename.endswith('.jpg'):
                    with open(resource_path(r'Staging Area\\'+filename), 'rb') as f:
                        data = f.read()
                    attachment = MIMEBase('application', 'octet-stream')
                    attachment.set_payload(data)
                    encoders.encode_base64(attachment)
                    attachment.add_header('Content-Disposition', 'attachment', filename=filename)
                    msg.attach(attachment)
                          
            print('4')

    else:#If files does not contain STL
        if send_mail:
            msg = email.message.Message()
            msg['From'] = your_email_address
            msg['To'] =  sender
            msg['Subject'] = 'Quote Results'

            # set the email content
            msg.set_payload('PLease Send only STL Files.')

    if send_mail:  
        smtp = smtplib.SMTP('smtp.gmail.com', 587)
        smtp.starttls()
        smtp.login(your_email_address, your_password)
        smtp.sendmail(your_email_address, sender, msg.as_string())
        smtp.quit()
                                            
    for filename in os.listdir(resource_path(r'Staging Area\\')):
        file_path = os.path.join(resource_path(r'Staging Area\\'), filename)
        if os.path.isfile(file_path):
                os.unlink(file_path)