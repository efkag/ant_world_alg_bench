#include <cstdlib>

// BoB robotics includes
#include "common/main.h"
#include "common/timer.h"
#include "imgproc/roll.h"
#include "navigation/perfect_memory.h"
#include "navigation/perfect_memory_window.h"

// Third party includes
#include "third_party/csv.h"
#include "third_party/path.h"
#include "plog/Log.h"

using namespace BoBRobotics;
using namespace units::angle;
using namespace units::literals;

int bobMain(int argc, char **argv)
{
    Navigation::PerfectMemoryRotater<> pm(cv::Size(180, 25));
    const filesystem::path dataPath{argv[1]};
    const auto imagePath = dataPath.parent_path();
    {
        // Open Training CSV
        io::CSVReader<1> trainingCSV((dataPath / "training.csv").str());
        trainingCSV.read_header(io::ignore_extra_column, "Filename");

        // Read snapshot filenames from file
        std::string snapshotFile;
        while(trainingCSV.read_row(snapshotFile)) {
            LOGI << "Training on " << snapshotFile;

            // Load snapshot
            cv::Mat snapshot = cv::imread((imagePath / snapshotFile).str());
            assert(!snapshot.empty());

            // Convert to grayscale
            cv::cvtColor(snapshot, snapshot, cv::COLOR_BGR2GRAY);

            // Add to PM
            pm.train(snapshot);
        }
    }

    {
        double pm0TestTime = 0.0;
        const size_t numScanColumns = (size_t)std::round(turn_t(90_deg).value() * 180.0);

        // Open Testing CSV
        io::CSVReader<4> testingCSV((dataPath / "testing_pm0.csv").str());
        testingCSV.read_header(io::ignore_extra_column, "Best heading [degrees]", "Lowest difference", "Best snapshot index", "Filename");

        // Read test points from file
        double bestHeading;
        double lowestDifference;
        unsigned int bestSnapshotIndex;
        std::string snapshotFile;
        while(testingCSV.read_row(bestHeading, lowestDifference, bestSnapshotIndex, snapshotFile)) {
             // Load snapshot
            cv::Mat snapshot = cv::imread((dataPath / snapshotFile).str());
            assert(!snapshot.empty());

            // Convert to grayscale
            cv::cvtColor(snapshot, snapshot, cv::COLOR_BGR2GRAY);

            degree_t leftBestHeading;
            float leftLowestDifference;
            size_t leftBestSnapshot;
            degree_t rightBestHeading;
            float rightLowestDifference;
            size_t rightBestSnapshot;
            {
                TimerAccumulate<> t(pm0TestTime);

                // Get best heading from left side of scan
                std::tie(leftBestHeading, leftBestSnapshot, leftLowestDifference, std::ignore) = pm.getHeading(
                    snapshot, 1, 0, numScanColumns);

                // Get best heading from right side of scan
                std::tie(rightBestHeading, rightBestSnapshot, rightLowestDifference, std::ignore) = pm.getHeading(
                    snapshot, 1, 180 - numScanColumns, 180);
            }
            // If best result came from left scan
            if(leftLowestDifference < rightLowestDifference) {
                LOGI << "Lowest difference: " << (leftLowestDifference / 255.0f) << "(" << lowestDifference << "), Best heading:" << leftBestHeading.value() << "(" << bestHeading << "), Best snapshot: " << leftBestSnapshot << "(" << bestSnapshotIndex << ")";
            }
            else {
                LOGI << "Lowest difference: " << (rightLowestDifference / 255.0f) << "(" << lowestDifference << "), Best heading:" << rightBestHeading.value() << "(" << bestHeading << "), Best snapshot: " << rightBestSnapshot << "(" << bestSnapshotIndex << ")";
            }
        }

        std::cout << "PM0 - Test time:" << pm0TestTime << "ms" << std::endl;
    }

    {
        double smw0TestTime = 0.0;
        const size_t numScanColumns = (size_t)std::round(turn_t(90_deg).value() * 180.0);

        // Open Testing CSV
        io::CSVReader<6> testingCSV((dataPath / "testing_smw0.csv").str());
        testingCSV.read_header(io::ignore_extra_column, "Best heading [degrees]", "Lowest difference", "Best snapshot index", "Filename", "Window start", "Window end");

        // **TODO** correct window settings
        Navigation::PerfectMemoryWindow::DynamicBestMatchGradient::WindowConfig windowConfig;
        Navigation::PerfectMemoryWindow::DynamicBestMatchGradient window(10, windowConfig);

        // Read test points from file
        double bestHeading;
        double lowestDifference;
        unsigned int bestSnapshotIndex;
        unsigned int windowStart;
        unsigned int windowEnd;
        std::string snapshotFile;
        while(testingCSV.read_row(bestHeading, lowestDifference, bestSnapshotIndex, snapshotFile, windowStart, windowEnd)) {
             // Load snapshot
            cv::Mat snapshot = cv::imread((dataPath / snapshotFile).str());
            assert(!snapshot.empty());

            // Convert to grayscale
            cv::cvtColor(snapshot, snapshot, cv::COLOR_BGR2GRAY);

            // Constrain current window
            const auto constrainedWindow = window.getWindow(pm.getNumSnapshots());

            degree_t leftBestHeading;
            float leftLowestDifference;
            size_t leftBestSnapshot;
            degree_t rightBestHeading;
            float rightLowestDifference;
            size_t rightBestSnapshot;
            {
                TimerAccumulate<> t(smw0TestTime);

                // Get best heading from left side of scan
                std::tie(leftBestHeading, leftBestSnapshot, leftLowestDifference, std::ignore) = pm.getHeading(
                    snapshot, ImgProc::Mask{}, constrainedWindow, 1, 0, numScanColumns);

                // Get best heading from right side of scan
                std::tie(rightBestHeading, rightBestSnapshot, rightLowestDifference, std::ignore) = pm.getHeading(
                    snapshot, ImgProc::Mask{}, constrainedWindow, 1, 180 - numScanColumns, 180);
            }

            // If best result came from left scan
            if(leftLowestDifference < rightLowestDifference) {
                LOGI << "Lowest difference: " << (leftLowestDifference / 255.0f) << "(" << lowestDifference << "), Best heading:" << leftBestHeading.value() << "(" << bestHeading << "), Best snapshot: " << leftBestSnapshot << "(" << bestSnapshotIndex << "), Window start: " << constrainedWindow.first << "(" << windowStart << "), Window end: " << constrainedWindow.second << "(" << windowEnd << ")";

                window.updateWindow(leftBestSnapshot, leftLowestDifference);
            }
            else {
                LOGI << "Lowest difference: " << (rightLowestDifference / 255.0f) << "(" << lowestDifference << "), Best heading:" << rightBestHeading.value() << "(" << bestHeading << "), Best snapshot: " << rightBestSnapshot << "(" << bestSnapshotIndex << "), Window start: " << constrainedWindow.first << "(" << windowStart << "), Window end: " << constrainedWindow.second << "(" << windowEnd << ")";

                window.updateWindow(rightBestSnapshot, rightLowestDifference);
            }
        }

        std::cout << "SMW0 - Test time:" << smw0TestTime << "ms" << std::endl;
    }

    return EXIT_SUCCESS;
}
