package main

import (
	"fmt"
	"os"
	"path/filepath"
	"runtime"

	ort "github.com/yalue/onnxruntime_go"
	"github.com/openvpi/game-infer/cmd"
)

func main() {
	libName := "libonnxruntime.dylib"
	if runtime.GOOS == "windows" {
		libName = "onnxruntime.dll"
	}
	exe, _ := os.Executable()
	libPath := filepath.Join(filepath.Dir(exe), libName)
	if _, err := os.Stat(libPath); err != nil {
		fmt.Fprintf(os.Stderr,
			"Error: %s not found.\nPlace it in the same directory as game-infer.\n", libName)
		os.Exit(1)
	}
	ort.SetSharedLibraryPath(libPath)
	if err := ort.InitializeEnvironment(); err != nil {
		fmt.Fprintf(os.Stderr, "Failed to initialize onnxruntime: %v\n", err)
		os.Exit(1)
	}
	defer ort.DestroyEnvironment()
	cmd.Execute()
}
