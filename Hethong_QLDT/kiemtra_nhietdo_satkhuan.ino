void kiemtra_nhietdo_satkhuan()
{
  // Đo nhiệt độ cổ tay
  while ((sta_qua_nhiet == 2) && (sta_measure == 0))
  {
    kiemtra_nhietdo();
    sta_measure = 1;
    tat_den_xanhduong();
    /////////////////////////////////////////////////////
    // Nếu nhiệt độ bình thường
    if ((sta_qua_nhiet == 0) && (sta_measure == 1))
    {
      // Mở LED xanh
      bat_den_xanhla();

      // Raspi doc ban co nhiet do binh thuong + cam on / Gui tin hieu kem nhiet do
      String RPi_tempC = "raspi binh thuong: " + String(rounded_val_tempC);
      // Truyền nhiệt độ đã đo được sang RPi
//      Serial.print("RPi_tempC:");
      Serial.println(RPi_tempC);
      int cho_raspi_2 = 1;

      // Sat khuan
      if ((!digitalRead(cb_cotay) == 1) && (sta_bom == 0)) // lần đầu có tay đưa vào
      {
        digitalWrite(relay_bom, 1);
        //        Serial.println("Bom run!");
        delay(thoi_gian_bom);
        // Ngừng động cơ
        digitalWrite(relay_bom, 0);
        sta_bom = 1; // Đã bơm xong dung dịch sát khuẩn
        Serial.println("Bom stop");

        while ((!digitalRead(cb_cotay) == 1) && (sta_bom == 1)) // chờ tới khi rút tay ra
        { // Do nothing
          //          Serial.println("In while nothing loop/ sta_bom:");
          //          Serial.println(sta_bom);
        }

        sta_bom = 0;
        //        Serial.print("Da thoat ra/ sta_bom: ");
        //        Serial.println(sta_bom);
      }

      // Cho doc cho xong de hoan tat chuong trinh
      while (cho_raspi_2 == 1)
      {
        if (Serial.available())
        {
          inputString = Serial.readString();
          inputString.toLowerCase();
          //            Serial.print("Tin hieu nhan:   ");
          //            Serial.println(inputString);
          if (inputString.startsWith("raspi da doc binh thuong"))
          {
            //*** Hoan tat chu trinh binh thuong
            delay(500);
            tat_den_xanhla();
            sta_bom = 0;
            cho_raspi_2 = 0;
            while ((!digitalRead(cb_nguoi) == 1) && (cho_raspi_2 == 0)) // chờ tới khi nguoi di ra
            { // Do nothing
            }
            Serial.println("ard_finish");
            break;
          }
        }
      }
    }

    ////////////////////////////////////////////////////////////////////////
    // Neu qua nhiet do
    if ((sta_qua_nhiet == 1) && (sta_measure == 1))
    {
      //Mo LED do
      canhbao_do();
      // Raspi doc ban co dau hieu sot / Gui tin hieu kem nhiet do
      String RPi_tempC = "raspi qua nhiet: " + String(rounded_val_tempC);
      // Truyền nhiệt độ đã đo được sang RPi
//      Serial.print("RPi_tempC:");
      Serial.println(RPi_tempC);
      int cho_raspi_3 = 1;
      // Sát khuẩn
      if ((!digitalRead(cb_cotay) == 1) && (sta_bom == 0)) // lần đầu có tay đưa vào
      {
        digitalWrite(relay_bom, 1);
        //        Serial.println("Bom run!");
        delay(thoi_gian_bom);
        // Ngừng động cơ
        digitalWrite(relay_bom, 0);
        sta_bom = 1; // Đã bơm xong dung dịch sát khuẩn
        //          Serial.println("Bom stop");

        while ((!digitalRead(cb_cotay) == 1) && (sta_bom == 1)) // chờ tới khi rút tay ra
        { // Do nothing
          //          Serial.println("In while nothing loop/ sta_bom:");
          //          Serial.println(sta_bom);
        }

        sta_bom = 0;
        //        Serial.print("Da thoat ra/ sta_bom: ");
        //        Serial.println(sta_bom);
      }

      // Cho doc xong và cho nhan button done
      while (cho_raspi_3 == 1)
      {
        if (Serial.available())
        {
          inputString = Serial.readString();
          inputString.toLowerCase();
          //            Serial.print("Tin hieu nhan:   ");
          //            Serial.println(inputString);
          if (inputString.startsWith("raspi da doc canh bao"))
          {
            //*** Hoan tat chu trinh binh thuong
            delay(500);
            // Cho nhan nut nhan done de hoan tat chuong trinh
            int cho_nhan_button_done = 1;
            while (cho_nhan_button_done == 1)
            {
              if (!digitalRead(button_done) == 1)
              {
                tat_canhbao_do();
                cho_nhan_button_done = 0;
                cho_raspi_3 = 0;
                Serial.println("ard_finish");
                while ((!digitalRead(cb_nguoi) == 1) && (cho_nhan_button_done == 0)) // chờ tới khi nguoi di ra
                { // Do nothing
                }
                break;
              }
            }
          }
        }
      }
    }
  }
}
