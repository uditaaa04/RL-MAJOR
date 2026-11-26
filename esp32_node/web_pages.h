// Web pages served by the ESP32 (live readings and the Wi-Fi setup page).
// They live in this header, not in esp32_node.ino, on purpose: the Arduino build scans the .ino for C++ functions
// to declare, and its old scanner mistakes the JavaScript in these pages (e.g. "function sc(){...}") for C++.
// Headers are not scanned, so keep any HTML/JavaScript here.
#pragma once

static const char PAGE_HEAD[] = R"HTML(<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Smart Conveyor QC - ESP32</title><style>
:root{--bg:#000;--fg:#fff;--soft:rgba(255,255,255,.6);--rule:rgba(255,255,255,.16);--cy:#00e0ff;--red:#ff3b30}
:root.light{--bg:#fff;--fg:#000;--soft:rgba(0,0,0,.6);--rule:rgba(0,0,0,.2);--cy:#007f99;--red:#d6261b}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.45 system-ui,Segoe UI,Roboto,sans-serif}
main{max-width:560px;margin:0 auto;padding:18px}h1{font-size:22px;margin:0 0 4px}p{color:var(--soft);margin:4px 0 14px}
.row{display:flex;justify-content:space-between;gap:12px;padding:9px 0;border-bottom:1px solid var(--rule)}
.row b{font-weight:600}.led{display:inline-block;width:11px;height:11px;border-radius:50%;background:var(--rule);margin-right:8px}
.led.on{background:var(--cy);box-shadow:0 0 10px var(--cy)}.bad{color:var(--red)}a{color:var(--cy)}
input,select,button{font:inherit;width:100%;box-sizing:border-box;padding:10px;margin:6px 0;border:1px solid var(--rule);
border-radius:6px;background:var(--bg);color:var(--fg)}button{background:var(--cy);color:#000;border:0;font-weight:600;cursor:pointer}
button.plain{background:none;color:var(--fg);border:1px solid var(--rule)}label{display:block;margin-top:10px;color:var(--soft)}
.top{display:flex;justify-content:space-between;align-items:center}.top button{width:auto;padding:6px 10px}
</style></head><body><main><div class="top"><h1>Smart Conveyor QC</h1><button class="plain" id="th">Light</button></div>
<script>var L=localStorage.getItem("t")==="l";function T(){document.documentElement.className=L?"light":"";
document.getElementById("th").textContent=L?"Dark":"Light"}T();document.getElementById("th").onclick=function(){L=!L;
localStorage.setItem("t",L?"l":"d");T()};</script>)HTML";

static const char PAGE_LIVE[] = R"HTML(<p>ESP32 edge node, live (3 times a second).</p>
<div class="row"><span><i class="led" id="c"></i>IR sensor 1 (camera)</span><b id="cn">-</b></div>
<div class="row"><span><i class="led" id="a"></i>IR sensor 2 (reject arm)</span><b id="an">-</b></div>
<div class="row"><span>Belt motor</span><b id="belt">-</b></div><div class="row"><span>Reject arm pushes</span><b id="pu">-</b></div>
<div class="row"><span>Gateway laptop (USB)</span><b id="usb">-</b></div><div class="row"><span>Tiny model loaded</span><b id="mo">-</b></div>
<div class="row"><span>Wi-Fi</span><b id="wf">-</b></div><div class="row"><span>Cloud dashboard</span><b id="cl">-</b></div>
<div class="row"><span>Running for</span><b id="up">-</b></div><div class="row"><span>Free memory</span><b id="me">-</b></div>
<p style="margin-top:16px"><a id="dash" href="#" hidden>Open the cloud dashboard</a> &nbsp; <a href="/wifi">Wi-Fi settings</a></p>
<script>function $(i){return document.getElementById(i)}function g(){fetch("/api/live").then(function(r){return r.json()}).then(function(d){
$("c").className="led"+(d.ir_camera?" on":"");$("a").className="led"+(d.ir_arm?" on":"");$("cn").textContent=d.count_camera+" boxes";
$("an").textContent=d.count_arm+" boxes";$("belt").textContent=d.duty>0?"running, "+Math.round(d.duty/2.55)+"%":"stopped";
$("pu").textContent=d.pushes;$("usb").textContent=d.usb_link?"connected":"not connected";$("mo").textContent=d.model?"yes":"no";
$("wf").textContent=d.wifi==="ONLINE"?d.ssid+", "+d.ip+" ("+d.rssi+" dBm)":d.wifi.toLowerCase()+(d.hotspot?", hotspot "+d.hotspot_name+" on":"");
$("cl").textContent=d.cloud;$("cl").className=/^ok/.test(d.cloud)?"":"bad";var s=Math.floor(d.uptime_ms/1000);
$("up").textContent=Math.floor(s/3600)+" h "+Math.floor(s%3600/60)+" min "+s%60+" s";$("me").textContent=Math.round(d.free_heap/1024)+" kB";
if(d.cloud_url){$("dash").href=d.cloud_url;$("dash").hidden=false}}).catch(function(){})}g();setInterval(g,330);</script>
</main></body></html>)HTML";

static const char PAGE_WIFI[] = R"HTML(<p>Wi-Fi for the ESP32 (2.4 GHz networks only). The board joins it and sends its readings to the cloud dashboard.</p>
<div class="row"><span>Now</span><b id="now">-</b></div>
<form method="POST" action="/save"><label>Wi-Fi name</label><select id="list"><option value="">Searching for networks ...</option></select>
<input name="ssid" id="ssid" placeholder="or type the name" required maxlength="32">
<label>Wi-Fi password</label><input name="pass" type="password" maxlength="63" placeholder="empty for an open network">
<label>Setup password (the hotspot password)</label><input name="pin" type="password" required>
<details><summary style="margin-top:12px;color:var(--soft)">Cloud server (normally set by the gateway over USB)</summary>
<label>Cloud address</label><input name="url" placeholder="https://name.onrender.com" maxlength="127">
<label>Cloud API key</label><input name="key" type="password" placeholder="leave empty to keep the current one" maxlength="79">
<label><input type="checkbox" name="insecure" value="1" style="width:auto"> skip the certificate check (testbed only)</label></details>
<button>Save and connect</button></form>
<form method="POST" action="/forget"><input name="pin" type="password" placeholder="setup password" required>
<button class="plain">Forget the saved Wi-Fi</button></form><p><a href="/">Live readings</a></p>
<script>function $(i){return document.getElementById(i)}$("list").onchange=function(){if(this.value)$("ssid").value=this.value};
function sc(){fetch("/api/scan").then(function(r){return r.json()}).then(function(d){if(!d.networks){setTimeout(sc,1500);return}
var l=$("list");l.innerHTML='<option value="">Pick a network ('+d.networks.length+' found)</option>';d.networks.forEach(function(n){
var o=document.createElement("option");o.value=n.ssid;o.textContent=n.ssid+"  ("+n.rssi+" dBm"+(n.open?", open":"")+")";l.appendChild(o)})})}sc();
function st(){fetch("/api/live").then(function(r){return r.json()}).then(function(d){$("now").textContent=d.wifi==="ONLINE"?
"online: "+d.ssid+", http://"+d.ip+" or http://"+d.host+".local":d.wifi.toLowerCase()+(d.ssid&&d.ssid!=="-"?" ("+d.ssid+")":"")})}st();setInterval(st,1500);</script>
</main></body></html>)HTML";
