// Adapted from Muons-Data-Generation by Lim Li Xin Jed and Qiu Ziming (2023).
// Apache-2.0; see third_party/Muons-Data-Generation-LICENSE and NOTICE.
// Unified integration changes: canonical voxel centres, validated I/O, headless
// operation, seeded source, exact outgoing plane, primary-only records and
// explicit return codes. This models ideal tracking planes, not RPC electronics.
#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "G4Box.hh"
#include "G4Event.hh"
#include "G4GeneralParticleSource.hh"
#include "G4LogicalVolume.hh"
#include "G4NistManager.hh"
#include "G4PVPlacement.hh"
#include "G4PhysListFactory.hh"
#include "G4RunManager.hh"
#include "G4SDManager.hh"
#include "G4Step.hh"
#include "G4SystemOfUnits.hh"
#include "G4Track.hh"
#include "G4UImanager.hh"
#include "G4VUserActionInitialization.hh"
#include "G4VUserDetectorConstruction.hh"
#include "G4VUserPrimaryGeneratorAction.hh"
#include "G4VSensitiveDetector.hh"
#include "G4Version.hh"
#include "Randomize.hh"

namespace {
constexpr int resolution = 64;
constexpr int voxelCount = resolution * resolution * resolution;
constexpr double targetHalfMm = 500.0;
constexpr double voxelWidthMm = 1000.0 / resolution;
constexpr double planeMm = 1000.0;
// 2 m x 2 m ideal acceptance, matching the legacy Mu-Net simulator geometry.
constexpr double detectorHalfMm = 1000.0;
const std::array<const char*, 22> materials = {
    "G4_AIR", "G4_BENZENE", "G4_METHANOL", "G4_WATER", "G4_Mg",
    "G4_CONCRETE", "G4_GYPSUM", "G4_Ca", "G4_S", "G4_Si", "G4_Al",
    "G4_Cs", "G4_Mn", "G4_Fe", "G4_I", "G4_Ni", "G4_Mo", "G4_Ag",
    "G4_Po", "G4_Pb", "G4_Au", "G4_U"
};

std::vector<int> readVoxels(const std::string& path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("Cannot open voxel file: " + path);
    std::vector<int> result;
    int label;
    while (input >> label) {
        if (label < 0 || label >= static_cast<int>(materials.size()))
            throw std::runtime_error("Voxel material ID outside 0..21");
        result.push_back(label);
        if (result.size() > voxelCount) throw std::runtime_error("Too many voxel IDs");
    }
    if (!input.eof() || result.size() != voxelCount)
        throw std::runtime_error("Expected exactly 64^3 integer voxel IDs in C/xyz order");
    return result;
}

class Detector : public G4VSensitiveDetector {
public:
    explicit Detector(std::ofstream& output) : G4VSensitiveDetector("outgoing_plane"), out(output) {}
    G4bool ProcessHits(G4Step* step, G4TouchableHistory*) override {
        auto* track = step->GetTrack();
        const auto* pre = step->GetPreStepPoint();
        if (track->GetParentID() != 0 || track->GetTrackID() != 1 ||
            track->GetParticleDefinition()->GetParticleName() != "mu-" ||
            pre->GetStepStatus() != fGeomBoundary) return true;
        const auto position = pre->GetPosition();
        const auto momentum = pre->GetMomentum();
        if (std::abs(position.z() / mm + planeMm) > 1e-4 || momentum.z() >= 0) return true;
        const auto initialPosition = track->GetVertexPosition();
        const auto initialDirection = track->GetVertexMomentumDirection();
        out << position.x()/mm << ',' << position.y()/mm << ',' << position.z()/mm << ','
            << momentum.x()/MeV << ',' << momentum.y()/MeV << ',' << momentum.z()/MeV << ','
            << initialPosition.x()/mm << ',' << initialPosition.y()/mm << ',' << initialPosition.z()/mm << ','
            << initialDirection.x() << ',' << initialDirection.y() << ',' << initialDirection.z() << ','
            << track->GetVertexKineticEnergy()/MeV << '\n';
        ++accepted;
        track->SetTrackStatus(fStopAndKill);
        return true;
    }
    long accepted = 0;
private:
    std::ofstream& out;
};

class Construction : public G4VUserDetectorConstruction {
public:
    Construction(std::vector<int> values, std::ofstream& stream, double sourceHalf)
        : labels(std::move(values)), out(stream), sourceHalfMm(sourceHalf) {}
    G4VPhysicalVolume* Construct() override {
        auto* nist = G4NistManager::Instance();
        auto* air = nist->FindOrBuildMaterial(materials[0]);
        const double worldHalf = std::max(sourceHalfMm, detectorHalfMm) + 200.0;
        auto* worldSolid = new G4Box("world", worldHalf*mm, worldHalf*mm, 1200*mm);
        auto* worldLogical = new G4LogicalVolume(worldSolid, air, "world");
        auto* world = new G4PVPlacement(nullptr, {}, worldLogical, "world", nullptr, false, 0);
        auto* voxel = new G4Box("voxel", voxelWidthMm*mm/2, voxelWidthMm*mm/2, voxelWidthMm*mm/2);
        std::array<G4LogicalVolume*, 22> logical{};
        for (int id = 1; id < static_cast<int>(materials.size()); ++id) {
            logical[id] = new G4LogicalVolume(voxel, nist->FindOrBuildMaterial(materials[id]),
                                             "material_" + std::to_string(id));
        }
        for (int flat = 0; flat < voxelCount; ++flat) {
            const int id = labels[flat];
            if (id == 0) continue;
            const int iz = flat % resolution;
            const int iy = (flat / resolution) % resolution;
            const int ix = flat / (resolution * resolution);
            // Cell centres match GridGeometry: lower + (index + 0.5) * width.
            const G4ThreeVector centre(
                (-targetHalfMm + (ix + 0.5)*voxelWidthMm)*mm,
                (-targetHalfMm + (iy + 0.5)*voxelWidthMm)*mm,
                (-targetHalfMm + (iz + 0.5)*voxelWidthMm)*mm);
            new G4PVPlacement(nullptr, centre, logical[id], "voxel", worldLogical, false, flat);
        }
        auto* planeSolid = new G4Box("outgoing_plane", detectorHalfMm*mm, detectorHalfMm*mm, 5*mm);
        detectorLogical = new G4LogicalVolume(planeSolid, air, "outgoing_plane");
        // Record pre-step position on top face at z = -1000 mm, not within slab.
        new G4PVPlacement(nullptr, G4ThreeVector(0, 0, -(planeMm+5)*mm), detectorLogical,
                          "outgoing_plane", worldLogical, false, 0);
        return world;
    }
    void ConstructSDandField() override {
        detector = new Detector(out);
        G4SDManager::GetSDMpointer()->AddNewDetector(detector);
        detectorLogical->SetSensitiveDetector(detector);
    }
    Detector* detector = nullptr;
private:
    std::vector<int> labels;
    std::ofstream& out;
    double sourceHalfMm;
    G4LogicalVolume* detectorLogical = nullptr;
};

class Generator : public G4VUserPrimaryGeneratorAction {
public:
    Generator() : source(std::make_unique<G4GeneralParticleSource>()) {}
    void GeneratePrimaries(G4Event* event) override { source->GeneratePrimaryVertex(event); }
private:
    std::unique_ptr<G4GeneralParticleSource> source;
};

class Actions : public G4VUserActionInitialization {
public:
    void Build() const override { SetUserAction(new Generator); }
};

long integerArgument(const char* value, const char* name) {
    std::size_t consumed = 0;
    const auto parsed = std::stol(value, &consumed);
    if (consumed != std::string(value).size() || parsed <= 0 || parsed > 2147483646L)
        throw std::runtime_error(std::string(name) + " must be an integer in 1..2147483646");
    return parsed;
}
}

int main(int argc, char** argv) {
    if (argc == 2 && std::string(argv[1]) == "--version") {
        std::cout << "muon-tomo-sim 1; " << G4Version << '\n';
        return 0;
    }
    if (argc != 7) {
        std::cerr << "Usage: muon_tomo_sim voxels.txt tracks.csv source.mac events seed source_half_mm\n";
        return 2;
    }
    try {
        const auto labels = readVoxels(argv[1]);
        const auto events = integerArgument(argv[4], "events");
        const auto seed = integerArgument(argv[5], "seed");
        std::size_t consumed;
        const double sourceHalf = std::stod(argv[6], &consumed);
        if (consumed != std::string(argv[6]).size() || !std::isfinite(sourceHalf) || sourceHalf <= 0)
            throw std::runtime_error("source_half_mm must be positive and finite");
        std::ifstream macro(argv[3]);
        if (!macro) throw std::runtime_error("Cannot open source macro");
        // Refuse to truncate an existing simulation result.
        if (std::ifstream(argv[2]).good()) throw std::runtime_error("Output CSV already exists");
        std::ofstream out(argv[2]);
        if (!out) throw std::runtime_error("Cannot open output CSV");
        out << std::setprecision(17);
        out << "out_x_mm,out_y_mm,out_z_mm,out_px_mev,out_py_mev,out_pz_mev,"
               "in_x_mm,in_y_mm,in_z_mm,in_dx,in_dy,in_dz,initial_kinetic_energy_mev\n";
        G4Random::setTheSeed(seed);
        auto manager = std::make_unique<G4RunManager>();
        auto* construction = new Construction(labels, out, sourceHalf);
        manager->SetUserInitialization(construction);
        G4PhysListFactory factory;
        manager->SetUserInitialization(factory.GetReferencePhysList("FTFP_BERT"));
        manager->SetUserInitialization(new Actions);
        manager->Initialize();
        auto* ui = G4UImanager::GetUIpointer();
        std::string line;
        while (std::getline(macro, line)) {
            if (line.empty() || line[0] == '#') continue;
            if (ui->ApplyCommand(line) != 0) throw std::runtime_error("Geant4 rejected: " + line);
        }
        manager->BeamOn(static_cast<int>(events));
        out.flush();
        if (!out) throw std::runtime_error("Failed writing output CSV");
        std::cout << "MUON_TOMO_SUMMARY generated=" << events
                  << " accepted=" << construction->detector->accepted
                  << " seed=" << seed << '\n';
        // Independent material X0 diagnostics; TFRecord labels retain the existing contract.
        std::cout << "MUON_TOMO_VOXEL_CENTRES_MM first=-492.1875 last=492.1875 width=15.625\n";
        for (int i = 1; i < static_cast<int>(materials.size()); ++i) {
            auto* mat = G4NistManager::Instance()->FindOrBuildMaterial(materials[i]);
            std::cout << "MUON_TOMO_MATERIAL " << i << ' ' << materials[i]
                      << " x0_cm=" << mat->GetRadlen()/cm << '\n';
        }
    } catch (const std::exception& exc) {
        std::cerr << "simulation error: " << exc.what() << '\n';
        return 1;
    }
    return 0;
}
