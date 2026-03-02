#include <Arduino.h>
#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>
#include <BLE2902.h>

// ===== Nordic UART UUID =====
#define UART_SERVICE_UUID        "6E400001-B5A3-F393-E0A9-E50E24DCCA9E"
#define UART_RX_CHAR_UUID        "6E400002-B5A3-F393-E0A9-E50E24DCCA9E" // PC -> ESP32
#define UART_TX_CHAR_UUID        "6E400003-B5A3-F393-E0A9-E50E24DCCA9E" // ESP32 -> PC

BLECharacteristic *txChar;
bool deviceConnected = false;

// ===== BLE Callbacks =====
class ServerCallbacks : public BLEServerCallbacks {
  void onConnect(BLEServer* pServer) {
    deviceConnected = true;
  }
  void onDisconnect(BLEServer* pServer) {
    deviceConnected = false;
    pServer->startAdvertising();
  }
};

// PC 寫資料進來（可選）
class RxCallbacks : public BLECharacteristicCallbacks {
  void onWrite(BLECharacteristic *pCharacteristic) {
    String v = pCharacteristic->getValue();   // ← 重點：用 String
    if (v.length()) {
      Serial.print("[BLE RX] ");
      Serial.println(v);
    }
  }
};

void setup() {
  Serial.begin(115200);
  delay(500);

  // ===== BLE Init =====
  BLEDevice::init("ESP32-EMG-Bridge");
  BLEServer *server = BLEDevice::createServer();
  server->setCallbacks(new ServerCallbacks());

  BLEService *service = server->createService(UART_SERVICE_UUID);

  txChar = service->createCharacteristic(
    UART_TX_CHAR_UUID,
    BLECharacteristic::PROPERTY_NOTIFY
  );
  txChar->addDescriptor(new BLE2902());

  BLECharacteristic *rxChar = service->createCharacteristic(
    UART_RX_CHAR_UUID,
    BLECharacteristic::PROPERTY_WRITE
  );
  rxChar->setCallbacks(new RxCallbacks());

  service->start();

  BLEAdvertising *adv = BLEDevice::getAdvertising();
  adv->addServiceUUID(UART_SERVICE_UUID);
  adv->start();

  Serial.println("[ESP32] BLE UART ready");
}

void loop() {
  // UART -> BLE
  if (deviceConnected && Serial.available()) {
    String line = Serial.readStringUntil('\n');
    if (line.length()) {
      line += "\n";
      txChar->setValue(line.c_str());
      txChar->notify();
    }
  }
}
