using System;
using System.Text;
using Newtonsoft.Json;
using ThermoFisher.CommonCore.Data;
using ThermoFisher.CommonCore.Data.Business;
using ThermoFisher.CommonCore.RawFileReader;

namespace RawFileReaderWrapper
{
    public class RawFileJson
    {
        public int first_spectrum;
        public int last_spectrum;
        public double[][] positions;
        public double[][] intensities;
        public double[] tms;
        public DateTime date;
    }

    class RawToJSON
    {
        const bool DEBUG = false;

        static void Main(string[] args)
        {
            RawFileJson rfjson = new RawFileJson();
            if(args.Length != 3)
            {
                Console.WriteLine("Usage: mono RawFileReaderWrapper.exe <raw_file> <outdir> <label>");
                return;
            }

            string raw_file = args[0];
            Console.WriteLine(raw_file);
            string outdir = args[1];
            string label = args[2];

            System.IO.Directory.CreateDirectory(outdir);

            using (var run = RawFileReaderAdapter.FileFactory(raw_file))
            {
                if (!run.IsOpen)
                {
                    Console.WriteLine(
                        String.Format("Failed to open {0} as raw file!", raw_file)
                    );
                    Console.WriteLine(run.FileError);
                    return;
                }

                if (DEBUG)
                {
                    Console.WriteLine("Number of instruments: " + run.InstrumentCount);
                    foreach (Device device in Enum.GetValues(typeof(Device)))
                    {
                        Console.Out.WriteLine(
                            Enum.GetName(typeof(Device), device) + ": " + run.GetInstrumentCountOfType(device)
                        );
                    }
                    Console.WriteLine("\nReading from MS...\n");
                }
                run.SelectInstrument(Device.MS, 1); //1 indexed

                var rh = run.RunHeader;

                if (DEBUG)
                {
                    Console.WriteLine("Start: " + rh.FirstSpectrum);
                    Console.WriteLine("End: " + rh.LastSpectrum);
                }

                int spectrums = rh.LastSpectrum - rh.FirstSpectrum;
                rfjson.first_spectrum = rh.FirstSpectrum;
                rfjson.last_spectrum = rh.LastSpectrum;
                rfjson.positions = new double[spectrums][];
                rfjson.intensities = new double[spectrums][];
                rfjson.tms = new double[spectrums];

                for (int i = rh.FirstSpectrum; i < rh.LastSpectrum; ++i)
                {
                    var scan = run.GetSegmentedScanFromScanNumber(i, null);
                    var intensities = scan.Intensities;
                    var positions = scan.Positions;
                    var tm = run.RetentionTimeFromScanNumber(i);

                    rfjson.positions[i-1] = positions;
                    rfjson.intensities[i-1] = intensities;
                    rfjson.tms[i-1] = tm;

                    if(DEBUG && i == 1)
                    {
                        const int n = 10;
                        Console.Out.WriteLine(String.Format("\nFirst {0} Positions:", n));
                        for (int j = 0; j < n; ++j)
                        {
                            Console.Out.WriteLine(positions[j]);
                        }
                        Console.Out.WriteLine(String.Format("\nFirst {0} Intensities:", n));
                        for (int j = 0; j < n; ++j)
                        {
                            Console.Out.WriteLine(intensities[j]);
                        }
                        Console.Out.WriteLine("\ntm:");
                        Console.Out.WriteLine(tm);
                    }
                }

                rfjson.date = run.FileHeader.CreationDate;
            }

            string output = JsonConvert.SerializeObject(rfjson);

            using (System.IO.FileStream outfile = System.IO.File.Open(System.IO.Path.Combine(outdir, label + ".json"), System.IO.FileMode.Create))
            {
                outfile.Write(Encoding.ASCII.GetBytes(output), 0, output.Length);
            }
        }
    }
}
