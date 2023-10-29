void test_den_xanhla()
{
  //  Serial.print("Test den xanh la: ");
  digitalWrite(relay_led_xanhla, 1);
  digitalWrite(led_xanhla, 1);
  //  Serial.print("ON  ||  ");
  delay(5000);
  digitalWrite(relay_led_xanhla, 0);
  digitalWrite(led_xanhla, 0);
  //
}
