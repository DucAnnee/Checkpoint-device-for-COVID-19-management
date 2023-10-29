void test_bom_satkhuan()
{
//    Serial.print("Test bom sat khuan: ");
  digitalWrite(relay_bom, 1);
  //  Serial.print("ON  ||  ");
  delay(5000);
  digitalWrite(relay_bom, 0);
  //  Serial.println("OFF");
}
