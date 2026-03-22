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
        // primarySwatch: Colors.deepPurple,
        scaffoldBackgroundColor: Colors.grey[200],
      ),
      home: Scaffold(
        appBar: AppBar(
          backgroundColor: Colors.white,
          title: const Text(
            "Veritas.ai",
            style: TextStyle(
              color: Colors.black,
              fontWeight: FontWeight.bold,
              fontSize: 25,
            ),
          ),
          centerTitle: true,
          elevation: 0,
        ),
        body: Center(
          child: Card(
            color: Colors.white,
            elevation: 10,
            shape: RoundedRectangleBorder(
              borderRadius: BorderRadius.circular(20),
            ),
            child: Padding(
              padding: const EdgeInsets.all(30),
              child: Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  Image.asset("assets/home.png", width: 150, height: 80),
                  const SizedBox(height: 5),

                  const Text(
                    "Veritas.ai",
                    style: TextStyle(fontSize: 24, fontWeight: FontWeight.bold),
                  ),

                  const SizedBox(height: 10),

                  const Text(
                    "Truth Over Illusion",

                    textAlign: TextAlign.center,
                    style: TextStyle(
                      color: Colors.grey,
                      fontFamily: 'Poppins',
                      fontSize: 15,
                    ),
                  ),

                  const SizedBox(height: 30),

                  SizedBox(
                    width: 220,
                    child: ElevatedButton.icon(
                      onPressed: startRecording,
                      icon: const Icon(Icons.select_all),
                      label: const Text(
                        "Choose App",
                        style: TextStyle(
                          color: Colors.black,
                          fontWeight: FontWeight.bold,
                          fontFamily: 'Poppins',
                        ),
                      ),
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
                      label: const Text(
                        "Stop Recording",
                        style: TextStyle(
                          color: Colors.black,
                          fontWeight: FontWeight.bold,
                          fontFamily: 'Poppins',
                        ),
                      ),
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
