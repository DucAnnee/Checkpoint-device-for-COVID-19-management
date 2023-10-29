void test_den_do()
{
  for (int i = 0; i <= 5; i++) {
  //  Serial.print("Test den do: ");
  digitalWrite(relay_led_do, 1);
  digitalWrite(led_do, 1);
  digitalWrite(buzzer, 1);
  //  Serial.print("ON  ||  ");
  delay(100);
  digitalWrite(relay_led_do, 0);
  digitalWrite(led_do, 0);
  digitalWrite(buzzer, 0);
  delay(100);
  //  Serial.println("OFF");
  }
  digitalWrite(relay_led_do, 1);
  digitalWrite(led_do, 1);
//  digitalWrite(buzzer, 1);
  //  Serial.print("ON  ||  ");
//  delay(100);
//  digitalWrite(relay_led_do, 0);
//  digitalWrite(led_do, 0);
////  digitalWrite(buzzer, 0);
//  delay(100);
}
