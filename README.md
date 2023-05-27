
# Manufacturing-Cost-Eastimation-Based-On-Deep-Learning:gear:

[![YouTube Video](https://img.shields.io/badge/YouTube-Play-red?logo=youtube)](https://www.youtube.com/channel/UC1CFjeCcq3rScWMtbaY6M2g)

![Methodology](https://github.com/Rutwik1000/Manufacturing-Cost-Eastimation-Based-On-Deep-Learning/assets/116753318/4b385b14-5072-4533-9adb-5db2a2e29196)




This project provides an automated solution for generating machining cost quotes based on client emails containing CAD files. :email:

The script continuously scans for new emails and performs the following steps:

1. :inbox_tray: **Email Scan**: The script actively monitors the specified email account for incoming messages.

2. :mag_right: **Quote Request Detection**: If a client sends an email with the subject "#quote" and attaches one or multiple STL files, the script identifies it as a quote request for CAD machining.

3. :arrow_down: **Download Files**: The attached STL files are automatically downloaded and stored in a designated "Staging Area" folder for processing.

4. :arrows_counterclockwise: **File Conversion**: The STL files are converted into the required binvox format using **SSDNeT** standards. This conversion prepares the files for predictions.

5. :gear: **Machining Feature Detection And Localization**: **SSDNeT** is employed to analyze the CAD files and predict the machining features performed. This step provides insights into the required manufacturing processes.

6. :moneybag: **Cost and Time Estimation**: Based on the predicted machining features and the volume of material to be machined, the script calculates the estimated cost and time required for the machining process.

7. :page_with_curl: **Quote Generation**: A detailed quotation is generated in PDF format, including the breakdown of costs, machining features, and project timeline.

8. :memo: **Record Logging**: The script updates a CSV log file with the filenames and unique identifiers (UID) of processed emails. This log ensures that previously processed emails are not reprocessed in the future.

9. :file_folder: **Archiving Files**: The received and generated files are compressed into a zip file and stored in a local **MySQL** server. This enables easy retrieval and access to historical data.

Explore the code and documentation to learn more about this automated machining cost quoting solution. :rocket:

