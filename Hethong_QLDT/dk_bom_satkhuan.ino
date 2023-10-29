void dk_bom_satkhuan()
{
  Serial.print("cb_cotay: ");
  Serial.print(digitalRead(cb_cotay));
  Serial.print(" || sta_bom: ");
  Serial.println(sta_bom); // =0
  if ((digitalRead(cb_cotay) == LOW) && (sta_bom == 0)) // lần đầu có tay đưa vào
  {
    digitalWrite(relay_bom, 1);
    Serial.println("Bom run!");
    delay(thoi_gian_bom);
    // Ngừng động cơ
    digitalWrite(relay_bom, 0);
    sta_bom = 1; // Đã bơm xong dung dịch sát khuẩn
    Serial.println("Bom stop");
    
    ////////////////////////////////////////////////////////////////
    // Đo nhiệt độ và gửi sang RPi cho xử lý trước khi chờ nhả tay//
    ////////////////////////////////////////////////////////////////
    
    while ((digitalRead(cb_cotay) == LOW) && (sta_bom == 1)) // chờ tới khi rút tay ra
    { // Do nothing
      Serial.println("In while nothing loop/ sta_bom:");
      Serial.println(sta_bom);
    }

    sta_bom = 0;
    Serial.print("Da thoat ra/ sta_bom: ");
    Serial.println(sta_bom);
  }
}
