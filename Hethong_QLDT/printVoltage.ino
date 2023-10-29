void printVoltage()
{
  int value = analogRead(voltageInputPin);// read the input
  float voltage =  value * (arduinoVoltage / 1023.0);//get the voltage from the value above
  Serial.print("Voltage: ");
  Serial.print(voltage);  
  Serial.println("V");
}
