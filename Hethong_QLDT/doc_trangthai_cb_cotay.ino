void doc_trangthai_cb_cotay()
{
//  Serial.println("Doc trang thai cam bien co tay");
  sta_cb_cotay = !digitalRead(cb_cotay);
  Serial.print(" Cam bien co tay: ");Serial.println(sta_cb_cotay); 
}
