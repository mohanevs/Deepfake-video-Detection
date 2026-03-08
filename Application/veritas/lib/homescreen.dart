import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

class Homescreen extends StatelessWidget {
  const Homescreen({super.key});

  static const platform = MethodChannel("screen_record_channel");

  Future<void> startRecording() async {
    await platform.invokeMethod("startRecording");
  }

  Future<void> stopRecording() async {
    final result = await platform.invokeMethod("stopRecording");
    print("Saved at: $result");
  }

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        primarySwatch: Colors.deepPurple,
        scaffoldBackgroundColor: const Color(0xFFF4F6FA),
      ),
      home: Scaffold(
        appBar: AppBar(
          title: const Text("Screen Recorder"),
          centerTitle: true,
          elevation: 0,
        ),
        body: Center(
          child: Card(
            elevation: 10,
            shape: RoundedRectangleBorder(
              borderRadius: BorderRadius.circular(20),
            ),
            child: Padding(
              padding: const EdgeInsets.all(30),
              child: Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  const Icon(
                    Icons.security,
                    size: 80,
                    color: Colors.deepPurple,
                  ),
                  const SizedBox(height: 20),

                  const Text(
                    "Veritas.ai",
                    style: TextStyle(fontSize: 24, fontWeight: FontWeight.bold),
                  ),

                  const SizedBox(height: 10),

                  const Text(
                    "Truth Over Illusion",
                    textAlign: TextAlign.center,
                    style: TextStyle(color: Colors.grey),
                  ),

                  const SizedBox(height: 30),

                  SizedBox(
                    width: 220,
                    child: ElevatedButton.icon(
                      onPressed: startRecording,
                      icon: const Icon(Icons.select_all),
                      label: const Text("Choose App"),
                      style: ElevatedButton.styleFrom(
                        padding: const EdgeInsets.symmetric(vertical: 14),
                        shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(12),
                        ),
                      ),
                    ),
                  ),

                  const SizedBox(height: 15),

                  SizedBox(
                    width: 220,
                    child: ElevatedButton.icon(
                      onPressed: stopRecording,
                      icon: const Icon(Icons.stop),
                      label: const Text("Stop Recording"),
                      style: ElevatedButton.styleFrom(
                        backgroundColor: Colors.orangeAccent,
                        padding: const EdgeInsets.symmetric(vertical: 14),
                        shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(12),
                        ),
                      ),
                    ),
                  ),
                ],
              ),
            ),
          ),
        ),
      ),
    );
  }
}
