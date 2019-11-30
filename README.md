# People Count
A module for counting people in a picture based on Yolo v.3 & S-DCNet

## Pre-requisite
- Install all the required packages listed in the `requirements.txt`
   ```
   $ pip install -r requirements.txt
   ```

- Download pre-trained weights file `weights.tar.gz`
   - [Google Drive](https://drive.google.com/open?id=1jU62jf8SbtsL73g0bvNt3vmwAVhzfv64)
   - Extract the downloaded weight file inside the project folder

## Test
- Run detect.py module with image_folder information
   ```
   $ python detect.py --image_folder /path/to/images
   ```

- To save result in a log file
   ```
   $ python detect.py --image_folder /path/to/images \
                      --log_enable 
   ```
   -- output.log file will be created and all the prediction result will be saved there

- To save result in a db
   ```
   $ python detect.py --image_folder /path/to/images \
                      --db_enable \
                      --db_host localhost \
                      --db_name DB_NAME \
                      --db_user_name DB_USER_NAME \
                      --db_user_pw DB_USER_PW \
                      --db_table DB_TABLE_NAME
   ```
   - Assuming the schema of the DB table as follows
   ```
   # Example Schema
   CREATE TABLE `count` (
                `filename` varchar(45) NOT NULL,
                `htag` varchar(45) DEFAULT NULL,
                `model` int(11) DEFAULT NULL,
                `result` varchar(255) DEFAULT NULL
                ) ENGINE=InnoDB DEFAULT CHARSET=latin1;

   ```
