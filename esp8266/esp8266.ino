#include <ESP8266HTTPClient.h>
#include <ESP8266WebServer.h>
#include <ESP8266WiFi.h>
#include <WiFiClient.h>

// ====================================================================
// Configuration: Load Wi-Fi credentials & Backend IP from secrets.h
// ====================================================================
#if __has_include("secrets.h")
  #include "secrets.h"
#else
  #warning "secrets.h not found! Using fallback placeholder credentials. Copy secrets.h.example to secrets.h."
  #define WIFI_SSID "YOUR_2.4GHZ_WIFI_SSID"
  #define WIFI_PASSWORD "YOUR_WIFI_PASSWORD"
  #define BACKEND_URL "http://192.168.1.100:8000/api/logs/"
#endif

const char *ssid = WIFI_SSID;
const char *password = WIFI_PASSWORD;
const char *backend_url = BACKEND_URL;


ESP8266WebServer server(80);

// Helper: send telemetry log event to FastAPI backend
void sendLogToBackend(String eventid, String ip, String username,
                      String password, String command) {
  if (WiFi.status() == WL_CONNECTED) {
    WiFiClient client;
    HTTPClient http;
    http.setTimeout(2000); // 2-second timeout to prevent blocking honeypot

    if (http.begin(client, backend_url)) {
      http.addHeader("Content-Type", "application/json");

      String jsonPayload = "{";
      jsonPayload += "\"eventid\":\"" + eventid + "\",";
      jsonPayload += "\"ip\":\"" + ip + "\",";
      jsonPayload += "\"username\":\"" + username + "\",";
      jsonPayload += "\"password\":\"" + password + "\",";
      jsonPayload += "\"commands\":[\"" + command + "\"]";
      jsonPayload += "}";

      int httpResponseCode = http.POST(jsonPayload);
      Serial.print("[BACKEND LOG] Sent: ");
      Serial.print(eventid);
      Serial.print(" | HTTP: ");
      Serial.println(httpResponseCode);

      http.end();
    }
  }
}

// 1. Root: Smart IoT Camera / Gateway login portal
void handleRoot() {
  String html =
      "<!DOCTYPE html><html><head><title>SecureMesh IoT Gateway</title>";
  html +=
      "<style>body{font-family:Arial;background:#1e293b;color:#f8fafc;display:"
      "flex;justify-content:center;align-items:center;height:100vh;margin:0;}";
  html += ".card{background:#0f172a;padding:30px;border-radius:10px;box-shadow:"
          "0 4px 6px rgba(0,0,0,0.3);width:320px;}";
  html += "h2{color:#38bdf8;margin-top:0;}input{width:100%;padding:10px;margin:"
          "8px 0;border-radius:5px;border:1px solid "
          "#334155;background:#1e293b;color:#fff;}";
  html +=
      "button{width:100%;padding:10px;background:#0284c7;border:none;color:#"
      "fff;font-weight:bold;border-radius:5px;cursor:pointer;margin-top:10px;}";
  html += "</style></head><body>";
  html += "<div class='card'><h2>IoT Device Portal</h2>";
  html += "<p style='color:#94a3b8;font-size:13px;'>Hardware: ESP8266-EX | "
          "Firmware: v1.0.4-sce</p>";
  html += "<form action='/login' method='POST'>";
  html += "<label>Username</label><input type='text' name='username' "
          "placeholder='admin'>";
  html += "<label>Password</label><input type='password' name='password' "
          "placeholder='••••••••'>";
  html += "<button type='submit'>Sign In</button>";
  html += "</form></div></body></html>";
  server.send(200, "text/html", html);
}

// 2. Authentication Trap: Logs brute-force credentials
void handleLogin() {
  String attacker_ip = server.client().remoteIP().toString();
  String username =
      server.hasArg("username") ? server.arg("username") : "empty";
  String password =
      server.hasArg("password") ? server.arg("password") : "empty";

  Serial.println("\n--- [SECURITY ALERT] Unauthorized Login Attempt ---");
  Serial.print("Attacker IP : ");
  Serial.println(attacker_ip);
  Serial.print("Username    : ");
  Serial.println(username);
  Serial.print("Password    : ");
  Serial.println(password);

  sendLogToBackend("esp8266.login.failed", attacker_ip, username, password, "");

  // Always return 401 to keep honeypot trapping attackers
  server.send(401, "application/json",
              "{\"error\": \"Invalid username or password\"}");
}

// 3. Command Injection Trap (/cgi-bin/status?cmd=...)
void handleCgiStatus() {
  String attacker_ip = server.client().remoteIP().toString();
  String cmd = server.hasArg("cmd") ? server.arg("cmd") : "";

  Serial.println("\n--- [SECURITY ALERT] CGI Probing / Exploit Attempt ---");
  Serial.print("Attacker IP : ");
  Serial.println(attacker_ip);
  Serial.print("Payload/Cmd : ");
  Serial.println(cmd);

  sendLogToBackend("esp8266.exploit.attempt", attacker_ip, "", "", cmd);

  // Honeypot bait response
  if (cmd.length() > 0) {
    server.send(200, "text/plain", "uid=0(root) gid=0(root) groups=0(root)\n");
  } else {
    server.send(200, "text/plain", "ESP8266 System Status: OK\n");
  }
}

// 4. Device Telemetry / Health Endpoint
void handleStatus() {
  String json = "{";
  json += "\"device\":\"ESP8266\",";
  json += "\"ip\":\"" + WiFi.localIP().toString() + "\",";
  json += "\"free_heap\":" + String(ESP.getFreeHeap()) + ",";
  json += "\"uptime_ms\":" + String(millis()) + ",";
  json += "\"rssi\":" + String(WiFi.RSSI());
  json += "}";
  server.send(200, "application/json", json);
}

void setup() {
  Serial.begin(115200);
  delay(500);

  Serial.println("\n=============================================");
  Serial.println("   SecureMesh-SCE: Physical ESP8266 Honeypot ");
  Serial.println("=============================================");
  Serial.print("Connecting to Wi-Fi: ");
  Serial.println(ssid);

  WiFi.mode(WIFI_STA);
  WiFi.begin(ssid, password);

  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 30) {
    delay(500);
    Serial.print(".");
    attempts++;
  }

  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("\n[SUCCESS] Wi-Fi Connected!");
    Serial.print("ESP8266 IP Address : http://");
    Serial.println(WiFi.localIP());
    Serial.print("Reporting to Backend: ");
    Serial.println(backend_url);
  } else {
    Serial.println("\n[WARNING] Wi-Fi connection timed out. Please verify SSID "
                   "& Password.");
  }

  // Setup Honeypot Web Server Endpoints
  server.on("/", HTTP_GET, handleRoot);
  server.on("/login", HTTP_POST, handleLogin);
  server.on("/cgi-bin/status", handleCgiStatus);
  server.on("/status", HTTP_GET, handleStatus);

  server.begin();
  Serial.println("[INFO] HTTP Honeypot service active on port 80");
  Serial.println("=============================================\n");
}

void loop() { server.handleClient(); }
