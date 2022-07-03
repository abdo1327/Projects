#include <Wire.h>
#include <SD.h>
#include <SPI.h>
long accelX, accelY, accelZ;
float gForceX, gForceY, gForceZ;
File MyFile;
long gyroX, gyroY, gyroZ;
float rotX, rotY, rotZ;
int pinCS = 10;
int fileNumber = 1;
int value = 0;
void setup() {
  Serial.begin(115200);
  Wire.begin();
  setupMPU();
  pinMode(pinCS, OUTPUT);

  if (SD.begin())
  {
    Serial.println("SD card is ready to use.");
    MyFile = SD.open("Test.csv", FILE_WRITE);
  }
  else
  {
    Serial.println("SD card initialization failed");
    return;
  }
  Serial.println("Enter Number s to start followed by #iteration.");
}

void loop() {
  if (Serial.available() > 0)
  {
    value = Serial.read();
    if (value == 's')
    {
      value = Serial.parseInt();
      Serial.println("######### started #########");
      printData(value);
      Serial.println("######### Done #########");
    }
    else if(value == 'c')
    {
      Serial.println("File Closed.");
      MyFile.close(); // close the file
    }
    Serial.flush();
  }
}
void setupMPU() {
  Wire.beginTransmission(0b1101000); //This is the I2C address of the MPU (b1101000
//b1101001 for AC0 low/high datasheet sec. 9.2)
  Wire.write(0x6B); //Accessing the register 6B - Power Management (Sec. 4.28)
  Wire.write(0b00000000); //Setting SLEEP register to 0. (Required; see Note on p. 9)
  Wire.endTransmission();
  Wire.beginTransmission(0b1101000); //I2C address of the MPU
  Wire.write(0x1B); //Accessing the register 1B - Gyroscope Configuration (Sec. 4.4) 
  Wire.write(0x00000000); //Setting the gyro to full scale +/- 250deg./s 
  Wire.endTransmission();
  Wire.beginTransmission(0b1101000); //I2C address of the MPU
  Wire.write(0x1C); //Accessing the register 1C - Acccelerometer Configuration (Sec. 4.5) 
  Wire.write(0b00000000); //Setting the accel to +/- 2g
  Wire.endTransmission();
}
void recordAccelRegisters() {
  Wire.beginTransmission(0b1101000); //I2C address of the MPU (Bank, 2016; Inc., 2013)
  Wire.write(0x3B); //Starting register for Accel Readings
  Wire.endTransmission();
  Wire.requestFrom(0b1101000, 6); //Request Accel Registers (3B - 40)
  while (Wire.available() < 6);
  accelX = Wire.read() << 8 | Wire.read(); //Store first two bytes into accelX
  accelY = Wire.read() << 8 | Wire.read(); //Store middle two bytes into accelY
  accelZ = Wire.read() << 8 | Wire.read(); //Store last two bytes into accelZ
  processAccelData();
}

void processAccelData() {
  gForceX = accelX / 16384.0;
  gForceY = accelY / 16384.0;
  gForceZ = accelZ / 16384.0;
}
void recordGyroRegisters() {
  Wire.beginTransmission(0b1101000); //I2C address of the MPU
  Wire.write(0x43); //Starting register for Gyro Readings
  Wire.endTransmission();
  Wire.requestFrom(0b1101000, 6); //Request Gyro Registers (43 - 48)
  while (Wire.available() < 6);
  gyroX = Wire.read() << 8 | Wire.read(); //Store first two bytes into accelX
  gyroY = Wire.read() << 8 | Wire.read(); //Store middle two bytes into accelY
  gyroZ = Wire.read() << 8 | Wire.read(); //Store last two bytes into accelZ
  processGyroData();
}
void processGyroData() {
  rotX = gyroX / 131.0;
  rotY = gyroY / 131.0;
  rotZ = gyroZ / 131.0;
}
void printData(int samples)
{
  for (size_t i = 0; i < samples; i++)
  {
    recordAccelRegisters();
    recordGyroRegisters();
    serialDisplay();
    fileWrite();
  }
  MyFile.println("********");
  String s{ "Enter Number 1 to start. file #" };
  Serial.println(s.c_str());
}
void serialDisplay()  // to display the values on the serial monitor
{
  Serial.print("Gyro (deg)");
  Serial.print(",");
  Serial.print(" X=");
  Serial.print(",");
  Serial.print(rotX);
  Serial.print(",");
  Serial.print(" Y=");
  Serial.print(",");
  Serial.print(rotY);
  Serial.print(" Z=");
  Serial.print(",");
  Serial.print(rotZ);
  Serial.print(",");
  Serial.print(" Accel (g)");
  Serial.print(",");
  Serial.print(" X=");
  Serial.print(",");
  Serial.print(gForceX);
  Serial.print(",");
  Serial.print(" Y=");
  Serial.print(",");
  Serial.print(gForceY);
  Serial.print(",");
  Serial.print(" Z=");
  Serial.print(",");
  Serial.println(gForceZ);

}

void fileWrite() //  to write the values from the sensor on the SD card 
{
  if (MyFile) {
    MyFile.print("Gyro (deg) X");
    MyFile.print(",");
    MyFile.print(rotX);
    MyFile.print(",");
    MyFile.print(" Gyro (deg)Y=");
    MyFile.print(",");
    MyFile.print(rotY);
    MyFile.print(",");
    MyFile.print(" Gyro (deg)Z=");
    MyFile.print(",");
    MyFile.print(rotZ);
    MyFile.print(",");
    MyFile.print(" Accel (g)X=");
    MyFile.print(",");
    MyFile.print(gForceX);
    MyFile.print(",");
    MyFile.print("Accel (g) Y=");
    MyFile.print(",");
    MyFile.print(gForceY);
    MyFile.print(",");
    MyFile.print(" Accel (g) Z=");
    MyFile.print(",");
    MyFile.println(gForceZ);
  }
  // if the file didn't open, print an error:
  else {
    Serial.println("error opening file");
  }
}
