#include <ESP8266WiFi.h>
#include <ESP8266WebServer.h>
#include <ESP8266HTTPClient.h>
#include <WiFiClient.h>

const char* ssid = "YOUR_WIFI_SSID";
const char* password = "YOUR_WIFI_PASSWORD";

// Replace with your computer's IP address running the FastAPI backend
const char* backend_url = "http://192.168.1.XXX:8000/api/logs/";

ESP8266WebServer server(80);

void handleRoot() {
  String html = "<html><head><title>Smart Camera Login</title></head><body>";
  html += "<h2>Smart Camera Administration</h2>";
  html += "<form action=\"/login\" method=\"POST\">";
  html += "Username: <input type=\"text\" name=\"username\"><br>";
  html += "Password: <input type=\"password\" name=\"password\"><br>";
  html += "<input type=\"submit\" value=\"Login\">";
  html += "</form></body></html>";
  server.send(200, "text/html", html);
}

void handleLogin() {
  if (server.hasArg("username") && server.hasArg("password")) {
    String username = server.arg("username");
    String password = server.arg("password");
    String attacker_ip = server.client().remoteIP().toString();
    
    // Log attempt to backend
    sendLogToBackend(attacker_ip, username, password);
    
    // Always fail login for honeypot
    server.send(401, "text/plain", "Invalid Credentials");
  } else {
    server.send(400, "text/plain", "Bad Request");
  }
}

void sendLogToBackend(String ip, String username, String password) {
  if (WiFi.status() == WL_CONNECTED) {
    WiFiClient client;
    HTTPClient http;
    
    http.begin(client, backend_url);
    http.addHeader("Content-Type", "application/json");
    
    String jsonPayload = "{\"eventid\":\"esp8266.login.failed\", \"ip\":\"" + ip + "\", \"username\":\"" + username + "\", \"password\":\"" + password + "\", \"commands\":[]}";
    
    int httpResponseCode = http.POST(jsonPayload);
    Serial.print("HTTP Response code: ");
    Serial.println(httpResponseCode);
    
    http.end();
  }
}

void setup() {
  Serial.begin(115200);
  WiFi.begin(ssid, password);
  
  while (WiFi.status() != WL_CONNECTED) {
    delay(1000);
    Serial.print(".");
  }
  
  Serial.println("");
  Serial.print("Connected to WiFi network with IP Address: ");
  Serial.println(WiFi.localIP());

  server.on("/", HTTP_GET, handleRoot);
  server.on("/login", HTTP_POST, handleLogin);
  
  server.begin();
  Serial.println("HTTP server started");
}

void loop() {
  server.handleClient();
}
