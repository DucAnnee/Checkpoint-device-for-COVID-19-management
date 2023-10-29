void kiemtra_nhietdo()
{
  do_nhietdo_cotay();
  //  String RPi_tempC = "Temp" + rounded_val_tempC;
  //  // Truyền nhiệt độ đã đo được sang RPi
  //  Serial.println(RPi_tempC);
//  Serial.println(rounded_val_tempC);

  // Kiểm tra trạng thái thân nhiệt
  if ((rounded_val_tempC < 3000) && (rounded_val_tempC > 5500) ) sta_qua_nhiet = 2;
  if (rounded_val_tempC <= 3750) sta_qua_nhiet = 0;
  if (rounded_val_tempC > 3750) sta_qua_nhiet = 1;
//  Serial.print("sta_qua_nhiet = ");
//  Serial.println(sta_qua_nhiet);
}
