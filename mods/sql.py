import mysql.connector
import os
import zipfile

def to_sqldb(uid,filenames):
    # define the database connection parameters
    host = "localhost"
    user = "root"
    password = "1234"

    # create a connection to the database
    cnx = mysql.connector.connect(host=host, user=user, passwd=password, database="logdb")

    # create a cursor object to execute SQL queries
    cursor = cnx.cursor()

    # define a list of file extensions to compress
    extensions = ['.stl', '.binvox', '.pdf', '.jpg']

    # create a ZIP file to store the compressed files
    zip_filename = str(uid)+'.zip'
    zip_path = os.path.join('Staging Area/', zip_filename)
    with zipfile.ZipFile(zip_path, 'w') as zip_file:
        # loop through each file in the current directory
        for filename in os.listdir('Staging Area/'):
            # get the file extension
            extension = os.path.splitext(filename)[1]

            # check if the file extension is in the list of extensions to include
            if extension in extensions:
                # add the file to the ZIP file
                zip_file.write(os.path.join('Staging Area/', filename), filename)

    # read the contents of the compressed file
    with open(zip_path, 'rb') as f:
        compressed_data = f.read()

    # insert the compressed file into the database
    query = "INSERT INTO files (uid,filenames, data) VALUES (%s, %s, %s)"
    values = (uid,str(filenames), compressed_data)
    cursor.execute(query, values)
    cnx.commit()

    # close the cursor and database connection
    cursor.close()
    cnx.close()