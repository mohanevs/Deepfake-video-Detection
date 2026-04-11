import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'dart:io';
import 'package:http/http.dart' as http;

class Homescreen extends StatefulWidget {
  const Homescreen({super.key});

  @override
  State<Homescreen> createState() => _HomescreenState();
}

class _HomescreenState extends State<Homescreen> {
  static const platform = MethodChannel("screen_record_channel");
  bool isRecording = false;
  bool isProcessing = false;
  String? statusMessage;
  List<Map<String, String>> installedApps = [];
  String? selectedAppPackage;
  String? selectedAppName;
  bool isLoadingApps = false;

  @override
  void initState() {
    super.initState();
    loadInstalledApps();
  }

  Future<void> startRecording() async {
    setState(() {
      isRecording = true;
      statusMessage = "Recording will start automatically...";
    });

    try {
      await platform.invokeMethod("startRecording");
    } catch (e) {
      setState(() {
        isRecording = false;
        statusMessage = "Error: ${e.toString()}";
      });
    }
  }

  Future<void> loadInstalledApps() async {
    setState(() {
      isLoadingApps = true;
    });

    try {
      final List<dynamic> apps = await platform.invokeMethod(
        'getInstalledApps',
      );
      setState(() {
        installedApps = apps.map((e) => Map<String, String>.from(e)).toList();
        isLoadingApps = false;
      });
    } catch (e) {
      setState(() {
        isLoadingApps = false;
        statusMessage = "Error loading apps: ${e.toString()}";
      });
    }
  }

  Future<void> stopRecording() async {
    try {
      await platform.invokeMethod("stopRecording");
      setState(() {
        isRecording = false;
        isProcessing = false;
        statusMessage = "Recording stopped";
      });

      ScaffoldMessenger.of(
        context,
      ).showSnackBar(const SnackBar(content: Text('Monitoring stopped')));

      // Clear status after 2 seconds
      await Future.delayed(const Duration(seconds: 2));
      if (mounted) {
        setState(() {
          statusMessage = null;
        });
      }
    } catch (e) {
      setState(() {
        statusMessage = "Error stopping: ${e.toString()}";
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        scaffoldBackgroundColor: const Color.fromARGB(255, 4, 38, 67),
      ),
      home: Scaffold(
        appBar: AppBar(
          backgroundColor: const Color.fromARGB(255, 4, 38, 67),
          title: const Text(
            "Veritas.ai",
            style: TextStyle(
              color: Colors.white,
              fontWeight: FontWeight.bold,
              fontSize: 25,
            ),
          ),
          centerTitle: true,
          elevation: 0,
        ),
        body: Center(
          child: SingleChildScrollView(
            padding: const EdgeInsets.all(16),
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
                      style: TextStyle(
                        fontSize: 24,
                        fontWeight: FontWeight.bold,
                      ),
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

                    // Record Button
                    SizedBox(
                      width: double.infinity,
                      child: ElevatedButton.icon(
                        onPressed:
                            (isRecording || isProcessing)
                                ? null
                                : startRecording,
                        icon: const Icon(Icons.play_arrow),
                        label: const Text(
                          "Choose App & Start Monitoring",
                          style: TextStyle(fontWeight: FontWeight.bold),
                        ),
                        style: ElevatedButton.styleFrom(
                          backgroundColor: Colors.green,
                          foregroundColor: Colors.white,
                          padding: const EdgeInsets.symmetric(vertical: 14),
                          shape: RoundedRectangleBorder(
                            borderRadius: BorderRadius.circular(12),
                          ),
                        ),
                      ),
                    ),

                    const SizedBox(height: 12),

                    // Stop Button
                    SizedBox(
                      width: double.infinity,
                      child: OutlinedButton.icon(
                        onPressed:
                            (isRecording || isProcessing)
                                ? stopRecording
                                : null,
                        icon: const Icon(Icons.stop),
                        label: const Text(
                          "Stop Monitoring",
                          style: TextStyle(fontWeight: FontWeight.bold),
                        ),
                        style: OutlinedButton.styleFrom(
                          foregroundColor: Colors.red,
                          side: const BorderSide(color: Colors.red),
                          padding: const EdgeInsets.symmetric(vertical: 14),
                          shape: RoundedRectangleBorder(
                            borderRadius: BorderRadius.circular(12),
                          ),
                        ),
                      ),
                    ),

                    const SizedBox(height: 20),

                    // Info Card
                    const SizedBox(height: 10),

                    Container(
                      padding: const EdgeInsets.all(12),
                      decoration: BoxDecoration(
                        color: Colors.blue.shade50,
                        borderRadius: BorderRadius.circular(10),
                      ),
                      child: const Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text(
                            " How it works:",
                            style: TextStyle(
                              fontWeight: FontWeight.bold,
                              fontSize: 14,
                            ),
                          ),
                          SizedBox(height: 8),
                          Text("1.Tap Choose App & Start Monitoring"),
                          Text("2. Select an app from the dropdown"),
                          Text(
                            "3. Grant screen recording permission when prompted",
                          ),
                          Text("4. 'LIVE' indicator appears on screen"),
                          Text("5. App records 20-second segments"),
                          Text("6. AI analyzes for deepfake detection"),
                          Text("7. Results show as REAL ✓ or FAKE ✗"),
                        ],
                      ),
                    ),
                  ],
                ),
              ),
            ),
          ),
        ),
      ),
    );
  }
}
