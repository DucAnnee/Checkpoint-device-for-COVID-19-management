void doc_trangthai_cb_nguoi()
{
//  Serial.println("Doc trang thai cam bien nguoi");
  sta_cb_nguoi = !digitalRead(cb_nguoi);
  Serial.print(" Cam bien nguoi: ");Serial.println(sta_cb_nguoi); 
}
