const express = require("express");
const http = require("http");
const { Server } = require("socket.io");
const cors = require("cors");

const app = express();
const server = http.createServer(app);
const io = new Server(server, {
  cors: { origin: "*" },
});

app.use(cors());
app.use(express.json());

// Python detector POSTs here when it detects a weapon
app.post("/detection", (req, res) => {
  const event = {
    label: req.body.label || "WEAPON",
    confidence: req.body.confidence || 0,
    timestamp: new Date().toISOString(),
  };
  io.emit("detection", event);
  res.json({ ok: true });
});

// Health check
app.get("/", (req, res) => res.json({ status: "running" }));

io.on("connection", (socket) => {
  console.log("Client connected:", socket.id);
});

const PORT = 4000;
server.listen(PORT, () => console.log(`Backend running on http://localhost:${PORT}`));
