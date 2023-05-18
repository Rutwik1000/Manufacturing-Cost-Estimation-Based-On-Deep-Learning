import pandas as pd
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
from reportlab.lib.pagesizes import letter, A4
import os
import numpy as np

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
 
    return os.path.join(base_path, relative_path)

def my_temp(c):
    c.translate(inch,inch)
    # define a large font
    c.setFont("Helvetica", 14)
    # choose some colors
    c.setStrokeColorRGB(0.1,0.8,0.1)
    c.setFillColorRGB(0,0,1) # font colour
    #c.drawImage('top2.jpg',-0.8*inch,9.3*inch)
    c.drawString(0, 9*inch, "Shop No : 1234, ABCD Road")
    c.drawString(0, 8.7*inch, "City Name: Mycity, ZIP : 12345")
    c.setFillColorRGB(0,0,0) # font colour
    c.line(0,8.6*inch,6.8*inch,8.6*inch)
    c.drawString(5.6*inch,9.5*inch,'Bill No :# 1234')
    from  datetime import date
    dt = date.today().strftime('%d-%b-%Y')
    c.drawString(5.6*inch,9.3*inch,dt)
    c.setFont("Helvetica", 8)
    c.drawString(3*inch,9.6*inch,'Tax No :# ABC1234')
    c.setFillColorRGB(1,0,0) # font colour
    c.setFont("Times-Bold", 40)
    c.drawString(4.3*inch,8.7*inch,'INVOICE')
    c.rotate(45) # rotate by 45 degree 
    c.setFillColorCMYK(0,0,0,0.08) # font colour CYAN, MAGENTA, YELLOW and BLACK
    c.setFont("Helvetica", 140) # font style and size
    c.drawString(2*inch, 1*inch, "SAMPLE") # String written 
    c.rotate(-45) # restore the rotation 
    c.setFillColorRGB(0,0,0) # font colour
    c.setFont("Times-Roman", 15)
    
    c.drawString(0*inch,8.3*inch,df.columns[0])
    c.drawString(0*inch,8.1*inch,units[0])
    
    c.drawString(0.5*inch,8.3*inch,df.columns[1])
    c.drawString(0.5*inch,8.1*inch,units[1])
    
    c.drawString(2*inch,8.3*inch,df.columns[2])
    c.drawString(2*inch,8.1*inch,units[2])
    
    c.drawString(3.5*inch,8.3*inch,df.columns[3])
    c.drawString(3.5*inch,8.1*inch,units[3])
    
    c.drawString(5*inch,8.3*inch,df.columns[4])
    c.drawString(5*inch,8.1*inch,units[4])
    
    c.drawString(6.1*inch,8.3*inch,df.columns[5])
    c.drawString(6.1*inch,8.1*inch,units[5])
    
    c.setStrokeColorCMYK(0,0,0,1) # vertical line colour 
    c.line(3.9*inch,8.3*inch,3.9*inch,2.7*inch)# first vertical line
    c.line(4.9*inch,8.3*inch,4.9*inch,2.7*inch)# second vertical line
    c.line(6.1*inch,8.3*inch,6.1*inch,2.7*inch)# third vertical line
    c.line(0.01*inch,2.5*inch,7*inch,2.5*inch)# horizontal line total
    
    c.drawString(1*inch,2.15*inch,'Sub-Total')
    c.drawString(1*inch,1.8*inch,'Discount')
    c.drawString(1*inch,1.2*inch,'Tax')
    c.setFont("Times-Bold", 22)
    c.drawString(2*inch,0.8*inch,'Total')
    c.setFont("Times-Roman", 22)
    c.drawString(5.6*inch,-0.1*inch,'Signature')
    c.setStrokeColorRGB(0.1,0.8,0.1) # Bottom Line colour 
    c.line(0,-0.7*inch,6.8*inch,-0.7*inch)
    c.setFont("Helvetica", 8) # font size
    c.setFillColorRGB(1,0,0) # font colour
    c.drawString(0, -0.9*inch, u"\u00A9"+" plus2net.com")
    
    return c

def get_pdf(df,units,filename):
    my_path=r'Staging Area/'+filename[:-4]+'.pdf' 
    discount_rate=10 # 10% discount 
    tax_rate=12 # tax rate  in percentage 

    c = canvas.Canvas(my_path,pagesize=letter)
    c=my_temp(c) # run the template

    c.setFillColorRGB(0,0,1) # font colour
    c.setFont("Helvetica", 20)
    row_gap=0.6 # gap between each row
    line_y=6.9 # location of fist Y position 
    total=0
    for index, items in df.iterrows():
        c.drawString(0*inch,line_y*inch,str(int(items[0]))) # p Name
        c.drawRightString(1.5*inch,line_y*inch,str(machtypes[int(items[1])])) # p Price
        c.drawRightString(2.5*inch,line_y*inch,str(items[2])) # p Qunt
        c.drawRightString(3.5*inch,line_y*inch,str(items[3])) # p Qunt
        c.drawRightString(5*inch,line_y*inch,str(items[4]))
        c.drawRightString(6.1*inch,line_y*inch,str(items[5]))# p Qunt


        sub_total=df['Cost'].sum()#my_prod[items][1]*my_sale[items]
        #c.drawRightString(7*inch,line_y*inch,str(sub_total)) # Sub Total 
        total=round(total+sub_total,1)
        line_y=line_y-row_gap


    c.drawRightString(7*inch,2.1*inch,str(float(total))) # Total 
    discount=round((discount_rate/100) * total,1)
    c.drawRightString(4*inch,1.8*inch,str(discount_rate)+'%') # discount
    c.drawRightString(7*inch,1.8*inch,'-'+str(discount)) # discount
    tax=round((tax_rate/100) * (total-discount),1)
    c.drawRightString(4*inch,1.2*inch,str(tax_rate)+'%') # tax 
    c.drawRightString(7*inch,1.2*inch,str(tax)) # tax 
    total_final=total-discount+tax
    c.setFont("Times-Bold", 22)
    c.setFillColorRGB(1,0,0) # font colour
    c.drawRightString(7*inch,0.8*inch,str(total_final)) # tax
    c.showPage()
    c.save()