"""Creating a link collision model for the Mycobot with a series of spheres of various radii
CATATAN PENTING:
- Semua posisi dalam frame link lokal (bukan COM frame)
- Posisi (x, y, z) dalam meter
- Radius dalam meter
- Index link mengikuti urutan URDF (0-5 untuk MyCobot 280)
"""

import numpy as np

# Link 0 (Base/Pundak)
# TIPS: Sesuaikan koordinat x,y,z berdasarkan geometri link di URDF
# ==============================================================================
# LINK 0 (Base Joint/Shoulder)
# ==============================================================================
# Dari data: Posisi world [0.1, 0, 0.2816]
# Link ini adalah base shoulder, memanjang vertikal ke atas
# Geometri: Silinder dengan tinggi ~0.10m, diameter ~0.08m

link_1_pos = (
    (0.0, 0.0, -0.11),   # Bola bawah (dekat base)
    (0.0, 0.0, -0.04),   # Bola tengah bawah
    # (0.0, 0.0, -0.02),   # Bola tengah atas
    # (0.0, 0.0, 0.00),   # Bola atas (dekat joint berikutnya)
)
link_1_radii = (0.057, 0.031, ) #0.031, 0.021

# ==============================================================================
# LINK 1 (Shoulder to Elbow / Upper Arm)
# ==============================================================================
# Dari data: Link frame di [0.1, 0, 0.2816], orientasi rotasi ~60°
# Bola saat ini bergerak di Y negatif → ini BENAR karena link memanjang
# Panjang link ~0.16m (dari posisi bola dunia yang terlihat)
# Perlu coverage lebih baik di sepanjang link

link_2_pos = (
    (0.0, 0.0, 0.00),   # Di joint
    (0.0, 0.0, 0.04),   # 1/4 panjang
    (0.0, 0.0, 0.065),   # 1/2 panjang
    (-0.055, 0.00, 0.065),   # Di Lengan 2
    # (0.0, 0.0, 0.158),  # Hampir di ujung (panjang link ~158mm)
)
link_2_radii = (0.029, 0.029, 0.029, 0.029)

# ==============================================================================
# LINK 2 (Elbow to Wrist / Forearm)
# ==============================================================================
# Dari data: Link frame di [0.1476, 0, 0.3812]
# Bola saat ini hanya 2, perlu lebih banyak untuk coverage
# Link memanjang ~0.14m

link_3_pos = (
    (0.0, 0.0, 0.00),   # Di elbow joint
    (0.0, 0.0, 0.035),  # 1/4 panjang
    (0.0, 0.0, 0.065),   # 1/2 panjang
    (-0.055, 0.0, 0.00),  # 3/4 panjang
    # (0.0, 0.0, 0.135),  # Mendekati wrist
)
link_3_radii = (0.028, 0.026, 0.026, 0.026)

# ==============================================================================
# LINK 3 (Wrist Pitch Joint)
# ==============================================================================
# Dari data: Link frame di [0.1951, -0.0646, 0.4646]
# Link kecil, perlu 2-3 bola untuk coverage

link_4_pos = (
    (0.0, 0.0, 0.00),   # Di joint
    # (0.0, 0.0, -0.025), # Tengah
    (0.0, 0.0, -0.054), # Ujung (panjang ~48mm)
)
link_4_radii = (0.025, 0.025)

# ==============================================================================
# LINK 4 (Wrist Roll Joint)
# ==============================================================================
# Dari data: Link frame di [0.2338, -0.0646, 0.5267]
# Link sangat kecil

link_5_pos = (
    (0.0, 0.0, 0.00),   # Di joint
    (0.0, 0.0, -0.06),  # Offset ke flange
    (0.0, -0.03, 0.0)
)
link_5_radii = (0.0215, 0.0215, 0.0215)

# ==============================================================================
# LINK 5 (Flange/End Effector Mount)
# ==============================================================================
# Dari data: Link frame di [0.2725, -0.0646, 0.5025]
# End effector mounting point

link_6_pos = (
    (0.0, 0.0, -0.01),    # Di flange
    (0.0, 0.0, -0.008),
    (0.0, 0.0, -0.013), # Sedikit ke belakang untuk coverage
)
link_6_radii = (0.0001, 0.0001, 0.057)

# Kumpulkan semua data dalam format tuple of tuples
# PENTING: Urutan harus sesuai dengan link index di URDF (0-5)
positions_list = (
    link_1_pos,
    link_2_pos,
    link_3_pos,
    link_4_pos,
    link_5_pos,
    link_6_pos,
)

radii_list = (
    link_1_radii,
    link_2_radii,
    link_3_radii,
    link_4_radii,
    link_5_radii,
    link_6_radii,
)

# Untuk backward compatibility
positions = {
    "link_1": link_1_pos,
    "link_2": link_2_pos,
    "link_3": link_3_pos,
    "link_4": link_4_pos,
    "link_5": link_5_pos,
    "link_6": link_6_pos,
}

radii = {
    "link_1": link_1_radii,
    "link_2": link_2_radii,
    "link_3": link_3_radii,
    "link_4": link_4_radii,
    "link_5": link_5_radii,
    "link_6": link_6_radii,
}

# Data yang digunakan oleh sistem
mycobot_collision_data = {
    "positions": positions_list, 
    "radii": radii_list
}

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def print_collision_model_info():
    """Print informasi model kolisi untuk debugging"""
    print("=" * 70)
    print("MYCOBOT 280 COLLISION MODEL INFO")
    print("=" * 70)
    
    total_spheres = 0
    for i, (pos, rad) in enumerate(zip(positions_list, radii_list)):
        print(f"\nLink {i} (joint{i+2}):")
        print(f"  Jumlah bola: {len(pos)}")
        
        if len(pos) > 0:
            print(f"  Posisi (link frame):")
            for j, (p, r) in enumerate(zip(pos, rad)):
                print(f"    Bola {j+1}: offset={p}, radius={r:.4f}m")
                total_spheres += 1
    
    print("\n" + "=" * 70)
    print(f"Total bola kolisi: {total_spheres}")
    print("=" * 70)


def get_coverage_stats():
    """Hitung statistik coverage untuk verifikasi"""
    stats = []
    
    for i, (positions, radii) in enumerate(zip(positions_list, radii_list)):
        if len(positions) == 0:
            continue
        
        # Hitung rentang Z coverage
        z_coords = [p[2] for p in positions]
        z_min, z_max = min(z_coords), max(z_coords)
        link_length = z_max - z_min if len(z_coords) > 1 else 0
        
        # Hitung rata-rata jarak antar bola
        if len(positions) > 1:
            distances = []
            for j in range(len(positions) - 1):
                dist = np.linalg.norm(
                    np.array(positions[j+1]) - np.array(positions[j])
                )
                distances.append(dist)
            avg_spacing = np.mean(distances)
        else:
            avg_spacing = 0
        
        stats.append({
            'link_idx': i,
            'num_spheres': len(positions),
            'link_length': link_length,
            'avg_radius': np.mean(radii),
            'avg_spacing': avg_spacing,
            'coverage_ratio': avg_spacing / (2 * np.mean(radii)) if avg_spacing > 0 else 0
        })
    
    return stats


if __name__ == "__main__":
    print_collision_model_info()
    
    print("\n" + "=" * 70)
    print("ANALISIS COVERAGE")
    print("=" * 70)
    
    stats = get_coverage_stats()
    for s in stats:
        print(f"\nLink {s['link_idx']}:")
        print(f"  Panjang link: {s['link_length']:.4f}m")
        print(f"  Jumlah bola: {s['num_spheres']}")
        print(f"  Rata-rata radius: {s['avg_radius']:.4f}m")
        print(f"  Jarak rata-rata antar bola: {s['avg_spacing']:.4f}m")
        print(f"  Coverage ratio (spacing/diameter): {s['coverage_ratio']:.2f}")
        
        # Evaluasi
        if s['coverage_ratio'] < 0.5:
            print("  ✅ Excellent overlap (>50%)")
        elif s['coverage_ratio'] < 0.8:
            print("  ✓ Good overlap (20-50%)")
        elif s['coverage_ratio'] < 1.0:
            print("  ⚠️ Minimal overlap (<20%)")
        else:
            print("  ❌ WARNING: Gap detected! Tambah bola atau perbesar radius")
    
    print("\n" + "=" * 70)