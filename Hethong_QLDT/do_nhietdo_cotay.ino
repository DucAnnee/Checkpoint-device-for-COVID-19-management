void do_nhietdo_cotay()
{
  tempC = mlx.readObjectTempC();
  //  tempC = mlx.readAmbientTempC();

  int value = analogRead(voltageInputPin);// read the input analog value
  float voltage =  value * (arduinoVoltage / 1023.0);//get the voltage from the value above
  //  Serial.println(voltage);
  val_tempC = tempC - (voltage - 3) * 0.6 + 4; // Bù nhiệt độ theo công thức từ Datasheet của cảm biến
  rounded_val_tempC = roundf(((val_tempC * 100) / 100) * 100);
  //  Serial.print(" Nhiet do co tay: "); Serial.println(val_tempC);
//  Serial.print(" Nhiet do lam tron: "); Serial.println(rounded_val_tempC);
//  return rounded_val_tempC;
}
