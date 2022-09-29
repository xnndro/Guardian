-- phpMyAdmin SQL Dump
-- version 5.2.0
-- https://www.phpmyadmin.net/
--
-- Host: localhost
-- Generation Time: Sep 25, 2022 at 03:15 PM
-- Server version: 10.4.21-MariaDB
-- PHP Version: 8.1.6

SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
START TRANSACTION;
SET time_zone = "+00:00";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8mb4 */;

--
-- Database: `guardians`
--

-- --------------------------------------------------------

--
-- Table structure for table `history`
--

CREATE TABLE `history` (
  `history_id` int(11) NOT NULL,
  `userID` varchar(6) NOT NULL,
  `comparingName` varchar(200) NOT NULL,
  `result` varchar(200) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

--
-- Dumping data for table `history`
--

INSERT INTO `history` (`history_id`, `userID`, `comparingName`, `result`) VALUES
(1, 'TI0334', 'Alexandro Alvin Valentino', 'MATCH');

-- --------------------------------------------------------

--
-- Table structure for table `karyawan`
--

CREATE TABLE `karyawan` (
  `namaKaryawan` varchar(200) NOT NULL,
  `pathfile` varchar(200) NOT NULL,
  `karyawanID` int(11) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

--
-- Dumping data for table `karyawan`
--

INSERT INTO `karyawan` (`namaKaryawan`, `pathfile`, `karyawanID`) VALUES
('Alexandro Alvin Valentino', '/Users/alexandroalvin/Documents/Guardians_FrontEnd/app/static/img/signatures/Alexandro Alvin Valentino', 1),
('Alleycia Syananda', '/Users/alexandroalvin/Documents/Guardians_FrontEnd/app/static/img/signatures/Alleycia Syananda', 2);

-- --------------------------------------------------------

--
-- Table structure for table `user`
--

CREATE TABLE `user` (
  `userID` varchar(6) NOT NULL,
  `NIP` varchar(4) DEFAULT NULL,
  `username` varchar(100) DEFAULT NULL,
  `passwords` varchar(50) DEFAULT NULL,
  `signature_registered` int(11) DEFAULT NULL,
  `signature_comparison` int(11) DEFAULT NULL,
  `document_verified` int(11) DEFAULT NULL,
  `is_admin` int(11) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

--
-- Dumping data for table `user`
--

INSERT INTO `user` (`userID`, `NIP`, `username`, `passwords`, `signature_registered`, `signature_comparison`, `document_verified`, `is_admin`) VALUES
('TI0334', '0334', 'Alexandro Alvin', 'xandroganteng', 2, 1, 1, 1),
('TI0335', '0335', 'Alleycia Syananda', 'alleyxixi', 2, 0, 0, 0),
('TI0343', '0343', 'Diva Amanda', 'divaputput', 2, 0, 0, 1),
('TI0348', '0348', 'Julius Salim', 'julius123', 2, 0, 0, 0);

--
-- Indexes for dumped tables
--

--
-- Indexes for table `history`
--
ALTER TABLE `history`
  ADD PRIMARY KEY (`history_id`),
  ADD KEY `fk_user` (`userID`);

--
-- Indexes for table `karyawan`
--
ALTER TABLE `karyawan`
  ADD PRIMARY KEY (`karyawanID`);

--
-- Indexes for table `user`
--
ALTER TABLE `user`
  ADD PRIMARY KEY (`userID`);

--
-- AUTO_INCREMENT for dumped tables
--

--
-- AUTO_INCREMENT for table `history`
--
ALTER TABLE `history`
  MODIFY `history_id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=2;

--
-- AUTO_INCREMENT for table `karyawan`
--
ALTER TABLE `karyawan`
  MODIFY `karyawanID` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=3;

--
-- Constraints for dumped tables
--

--
-- Constraints for table `history`
--
ALTER TABLE `history`
  ADD CONSTRAINT `fk_user` FOREIGN KEY (`userID`) REFERENCES `user` (`userID`) ON DELETE CASCADE ON UPDATE CASCADE;
COMMIT;

/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
