// Khai bao thu vien
#include <Wire.h>
#include <Adafruit_MLX90614.h>
Adafruit_MLX90614 mlx = Adafruit_MLX90614();

// Khai bao bien
String inputString; // Chuỗi dữ liệu đầu vào

//////////////////////////////////////////
// Khai bao chuong trinh con
void doc_trangthai_cb_nguoi();
void doc_trangthai_cb_cotay();
void do_nhietdo_cotay();
void kiemtra_nhietdo();
void kiemtra_nhietdo_satkhuan();
void printVoltage();

// Bơm sát khuẩn
void dk_bom_satkhuan();
void test_bom_satkhuan();

// Đèn đỏ
void test_den_do();
void canhbao_do();
void tat_canhbao_do();

// Đèn xanh lá
void bat_den_xanhla();
void tat_den_xanhla();
void test_den_xanhla();

// Đèn xanh dương
void bat_den_xanhduong();
void tat_den_xanhduong();
void test_den_xanhduong();

// Buzzer
void test_buzzer();

// Button
void doc_trangthai_button_done();
//////////////////////////////////////////

// 1. Khai bao giao tiep voi cam bien than nhiet
const int voltageInputPin = A0;
const float arduinoVoltage = 5.0; // Điện áp hoạt động của Arduino
float val_tempC, tempC;
int rounded_val_tempC;
int sta_measure = 0;
int sta_qua_nhiet = 2;

// 2. Khai bao cam bien
const int cb_nguoi = 2;
int sta_cb_nguoi = 0; // Trạng thái cảm biến người

const int cb_cotay = 3;
int sta_cb_cotay = 0; // Trạng thái cảm biến cổ tay

// 3. Khai bao bom sat khuan
const int relay_bom = 7;
int sta_bom = 0; // Trạng thái bơm sát khuẩn
int thoi_gian_bom = 300;

// 4. Khai bao relay Led day
const int relay_led_do = 4;
const int relay_led_xanhla = 5;
const int relay_led_xanhduong = 6;

// 5. Khai bao buzzer
const int buzzer = 8;

// 6. Khai bao Led
const int led_do = 9;
const int led_xanhla = 10;

// 7. Khai bao nut nhan done
const int button_done = 11;
int sta_button_done = 0;

void setup() {
  // 1. Khởi tạo cảm biến đo thân nhiệt
  mlx.begin();

  // 2. Khai bao chan cam bien
  pinMode(cb_nguoi, INPUT_PULLUP);
  pinMode(cb_cotay, INPUT_PULLUP);

  // 3. Khai bao bom sat khuan
  pinMode(relay_bom, OUTPUT);
  digitalWrite(relay_bom, LOW);

  // 4. Khai bao relay Led day
  pinMode(relay_led_do, OUTPUT);
  pinMode(relay_led_xanhla, OUTPUT);
  pinMode(relay_led_xanhduong, OUTPUT);
  digitalWrite(relay_led_do, LOW);// tat het led
  digitalWrite(relay_led_xanhla, LOW);
  digitalWrite(relay_led_xanhduong, LOW);

  // 5. Khai bao buzzer
  pinMode(buzzer, OUTPUT);
  digitalWrite(buzzer, LOW);

  // 6. Khai bao Led
  pinMode(led_do, OUTPUT);
  pinMode(led_xanhla, OUTPUT);
  digitalWrite(led_do, LOW);// tat het led
  digitalWrite(led_xanhla, LOW);

  // 7. Khai bao nut nhan done
  pinMode(button_done, INPUT_PULLUP);

  // Khởi tạo giao tiếp Serial
  Serial.begin(9600);
}

void loop() {
  //////////////////////////////////////////////////////////////////
  // CODE Chuong trinh chinh
  sta_cb_nguoi = !digitalRead(cb_nguoi);
//  Serial.print("sta_cb_nguoi: "); Serial.println(sta_cb_nguoi);
  if (!digitalRead(cb_nguoi) == 1) delay(2000); // Kiểm tra người vẫn còn đứng trước hệ thống sau 2 giây
  if (!digitalRead(cb_nguoi) == 1)
  {
    // Đã có người đến trước hệ thống, truyền "start_process" để RPi thực hiện chương trình
    // Bước 1: Chào mừng
    // Bước 2: Nhận dạng đeo khẩu trang
    // Bước 4: Chụp hình khuôn mặt
    // Bước 5.0: Cảnh báo vùng đỏ
    // Bước 5: Yêu cầu đo nhiệt độ
    // Truyền tín hiệu qua cho Arduino đo nhiệt độ, cấp dung dịch sát khuẩn và tiến hành cảnh báo, trả dữ liệu về cho Raspi (Bước 6)

    Serial.println("start_process");
    int cho_raspi_1 = 1; // Biến chờ tín hiệu "start_measure" từ RPi để tiến hành đo nhiệt độ (sau Bước 5 của Raspi)
    // Khi nào truyền Serial qua RPi thì mới kích hoạt biến chờ

    while (cho_raspi_1 == 1)
    {
      //      Serial.println("Dang cho RPi yeu cau thuc hien do nhiet do và sat khuan!");
      if (Serial.available())
      {
        inputString = Serial.readString();
        inputString.toLowerCase();
        //        Serial.print("Tin hieu nhan:   "); Serial.println(inputString);
        // RED
        if (inputString.startsWith("red_zone"))
        {
          canhbao_do();
          int cho_nhan_button_done = 1;
          while (cho_nhan_button_done == 1)
          {
            if (!digitalRead(button_done) == 1)
            {
              tat_canhbao_do();
              cho_nhan_button_done = 0;
              Serial.println("ard_finish");
              while ((!digitalRead(cb_nguoi) == 1) && (cho_nhan_button_done == 0)) // chờ tới khi nguoi di ra
              { // Do nothing
              }
              cho_raspi_1 = 0;
              break;
            }
//            Serial.println("Da toi day 7");
          }
//          Serial.println("Da toi day 8");
        }
        // BLUE
        if (inputString.startsWith("start_measure"))
        {
          // Bật đèn xanh dương và đèn trắng
          bat_den_xanhduong();
          int cho_cb_cotay = 1;
          while (cho_cb_cotay == 1)
          {
            sta_cb_cotay = !digitalRead(cb_cotay);
//            Serial.print("sta_cb_cotay: "); Serial.println(sta_cb_cotay);
            if (!digitalRead(cb_cotay) == 1) delay(1400); // Kiểm tra cổ tay vẫn còn để trước cảm biến sau 1 giây
            if (!digitalRead(cb_cotay) == 1)
            {
              //            Serial.print("cb_cotay: "); Serial.print(digitalRead(cb_cotay));
              // Serial.print(" || sta_measure: "); Serial.print(sta_measure); // =0
              // Serial.print(" || sta_bom: "); Serial.print(sta_bom); // =0
              if ((sta_measure == 0) && (sta_bom == 0)) // lần đầu có tay đưa vào
              {
                // Đo nhiệt độ cổ tay + sát khuẩn + cảnh báo độc lập với Raspi
                kiemtra_nhietdo_satkhuan();
                cho_raspi_1 = 0;
                cho_cb_cotay = 0;
                sta_measure = 0;
                sta_qua_nhiet = 2;
              }
//              Serial.println("Da toi day 1");
            }
//            Serial.println("Da toi day 2");
          }
//          Serial.println("Dang cho ccb_cotay");
        }
//        Serial.println("Da toi day 3");
      }
//      Serial.println("Da toi day 4");
    }
//    Serial.println("Da toi day 5");
  }
//  Serial.println("Da toi day 6");
  //////////////////////////////////////////////////////////////////
//    //  * Code Test phan cung
//      doc_trangthai_cb_nguoi(); // Xuống thấp là có người đến
//      doc_trangthai_cb_cotay(); // Xuống thấp là có cổ tay đưa vào
//      doc_trangthai_button_done();
//      do_nhietdo_cotay();
//      delay(500);
//  
//      while (Serial.available())
//      {
//        inputString = Serial.readString();
//        //    inputString = Serial.readStringUntil(' ');
//        inputString.toLowerCase();
//        Serial.print("Tin hieu nhan duoc qua Serial:   ");
//        Serial.println(inputString);
//        // ------------------------------------------------
//        if (inputString.startsWith("bom"))
//        {
//          test_bom_satkhuan();
//        }
//        // ------------------------------------------------
//        if (inputString.startsWith("do"))
//        {
//          test_den_do();
//        }
//        // ------------------------------------------------
//        if (inputString.startsWith("xl"))
//        {
//          test_den_xanhla();
//        }
//        // ------------------------------------------------
//        if (inputString.startsWith("xd"))
//        {
//          test_den_xanhduong();
//        }
//        // ----------------------------------------------
//        if (inputString.startsWith("buzzer"))
//        {
//          test_buzzer();
//        }
//        // ----------------------------------------------
//        if (inputString.startsWith("vol"))
//        {
//          printVoltage();
//        }
//      }
}
